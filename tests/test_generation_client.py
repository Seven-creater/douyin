"""generation S3 单测：minimax_client（manifest 语义 + 调度 mock）与 assemble 纯函数。"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from src.generation.assemble import (
    build_concat_cmd,
    build_drawtext_cmd,
    build_trim_cmd,
    escape_drawtext,
)
from src.generation.minimax_client import (
    ClipTask,
    GenerationManifest,
    GenerationScheduler,
    MiniMaxHTTP,
    expected_job_seconds,
)


# ---------- manifest ----------

def test_manifest_roundtrip_and_done(tmp_path):
    clips_root = tmp_path / "clips"
    m = GenerationManifest(tmp_path / "manifest.json", clips_root)
    m.record("v1", 0, status="submitted", job_id="j1")
    m.save()
    assert not m.is_done("v1", 0)                       # 没 done
    m.record("v1", 0, status="done")
    m.save()
    assert not m.is_done("v1", 0)                       # done 但文件不存在
    p = m.clip_path("v1", 0)
    p.parent.mkdir(parents=True)  # 含 clips/ 子目录
    p.write_bytes(b"x" * 20000)
    assert m.is_done("v1", 0)                            # done 且文件在
    # 重新加载持久化检查
    m2 = GenerationManifest(tmp_path / "manifest.json", clips_root)
    assert m2.get("v1", 0)["job_id"] == "j1"


def test_manifest_key_format(tmp_path):
    m = GenerationManifest(tmp_path / "m.json", tmp_path)
    assert m.key("giftbox", 3) == "giftbox/u03"
    assert m.clip_path("giftbox", 3).name == "u03.mp4"
    assert "clips" in str(m.clip_path("giftbox", 3))


# ---------- expected_job_seconds ----------

def test_expected_job_seconds_scales():
    assert expected_job_seconds(124) == pytest.approx(690 * 2 + 300)
    assert expected_job_seconds(248) > expected_job_seconds(124)


# ---------- FakeMiniMax 服务（http.server） ----------

class FakeMiniMaxHandler(BaseHTTPRequestHandler):
    jobs = {}
    next_state = ("queued", 0)  # 每次查询推进：(状态, 第几次查询后变 done)
    query_counts = {}

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_headers = None
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._json({"status": "ready"})
        if self.path.startswith("/jobs/"):
            jid = self.path.split("/")[-1]
            n = FakeMiniMaxHandler.query_counts.get(jid, 0) + 1
            FakeMiniMaxHandler.query_counts[jid] = n
            threshold, video = FakeMiniMaxHandler.next_state
            if n >= threshold:
                FakeMiniMaxHandler.jobs[jid]["status"] = "done"
                FakeMiniMaxHandler.jobs[jid]["video"] = video
            return self._json(FakeMiniMaxHandler.jobs[jid])
        self._json({}, 404)

    def do_POST(self):
        if self.path == "/generate":
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length))
            jid = f"j{len(FakeMiniMaxHandler.jobs) + 1}"
            FakeMiniMaxHandler.jobs[jid] = {"status": "queued", "request": req}
            return self._json({"job_id": jid, "status": "queued"}, 202)
        self._json({}, 404)


@pytest.fixture()
def fake_server(tmp_path):
    video = tmp_path / "gen_out.mp4"
    video.write_bytes(b"x" * 20000)
    FakeMiniMaxHandler.jobs = {}
    FakeMiniMaxHandler.query_counts = {}
    FakeMiniMaxHandler.next_state = (2, str(video))  # 第 2 次查询变 done
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeMiniMaxHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()


def _fake_svc_cfg(port):
    return {"host": "127.0.0.1", "base_port": port, "gpu_pairs": ["0,1"], "max_instances": 1,
            "poll_interval_s": 0, "job_timeout_factor": 2.0, "max_retries": 2}


def test_scheduler_end_to_end(fake_server, tmp_path, monkeypatch):
    port = fake_server.server_address[1]
    import src.generation.minimax_client as mc

    svc = mc.MiniMaxService({**_fake_svc_cfg(port)})
    monkeypatch.setattr(svc, "base_url", lambda p: f"http://127.0.0.1:{port}")
    monkeypatch.setattr(svc, "probe", lambda p: {"status": "ready"})

    clips_root = tmp_path / "clips"
    manifest = GenerationManifest(tmp_path / "manifest.json", clips_root)
    sched = GenerationScheduler([port], svc.http, manifest, svc.cfg, clips_root, width=544, height=960)
    task = ClipTask(variant_id="v1", unit_id=0, prompt="Vertical 9:16 ...", seed=42,
                    num_frames=124, output_name="t_v1_u00.mp4")
    stats = sched.run(svc, [task])
    assert manifest.is_done("v1", 0)
    assert (clips_root / "v1" / "clips" / "u00.mp4").stat().st_size == 20000
    entry = manifest.get("v1", 0)
    assert entry["status"] == "done" and entry["seed"] == 42
    assert stats["done"] >= 1


def test_scheduler_skips_done(fake_server, tmp_path, monkeypatch):
    port = fake_server.server_address[1]
    import src.generation.minimax_client as mc

    clips_root = tmp_path / "clips"
    manifest = GenerationManifest(tmp_path / "manifest.json", clips_root)
    p = clips_root / "v1" / "clips" / "u00.mp4"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"x" * 20000)
    manifest.record("v1", 0, status="done")
    manifest.save()
    svc = mc.MiniMaxService({**_fake_svc_cfg(port)})
    sched = GenerationScheduler([port], svc.http, manifest, svc.cfg, clips_root, 544, 960)
    task = ClipTask(variant_id="v1", unit_id=0, prompt="p", seed=1, num_frames=124, output_name="o.mp4")
    sched.run(svc, [task])
    assert FakeMiniMaxHandler.jobs == {}  # 完全没提交


def test_ensure_instances_caps_to_free_pairs(monkeypatch):
    """空闲卡对不足时实例数收敛到 free_pairs，且不重复分配卡对。"""
    import src.generation.minimax_client as mc

    svc = mc.MiniMaxService({"host": "127.0.0.1", "base_port": 8300,
                             "gpu_pairs": ["0,1", "2,3", "4,5", "6,7"],
                             "max_instances": 4, "health_timeout_s": 5})
    monkeypatch.setattr(svc, "preflight_gpus", lambda: (["2,3", "4,5", "6,7"], ["卡对 0,1 忙"]))
    started: list[tuple[int, str]] = []
    monkeypatch.setattr(svc, "start_instance",
                        lambda port, pair: started.append((port, pair)) or True)
    calls = {"n": 0}

    def fake_health(base_url):  # 首查未启动，启动后 ready
        calls["n"] += 1
        return {"status": "ready"} if calls["n"] % 2 == 0 else None

    monkeypatch.setattr(svc.http, "health", fake_health)
    ready, warns = svc.ensure_instances(4)
    assert len(started) == 3 and len(ready) == 3          # 4 → 3，不再抢卡
    pairs = [p for _, p in started]
    assert pairs == ["2,3", "4,5", "6,7"] and len(set(pairs)) == 3
    assert any("3" in w for w in warns)                   # 降容告警落账


# ---------- assemble 纯函数 ----------

def test_escape_drawtext():
    assert escape_drawtext("plain") == "plain"
    out = escape_drawtext("50%的:人'说,好")
    assert "\\%" in out and "\\:" in out and "," not in out.replace("\\,", "")
    assert "\\'" in out or "\\\\'" in out


def test_build_trim_cmd_shape(tmp_path):
    cmd = build_trim_cmd(tmp_path / "a.mp4", tmp_path / "b.mp4", ss=0.3, t=3.2, audio_rate=44100, crf=20)
    assert cmd[0] == "ffmpeg" and "-ss" in cmd and "0.3" in cmd and "3.2" in cmd


def test_build_concat_cmd(tmp_path):
    cmd = build_concat_cmd(tmp_path / "l.txt", tmp_path / "o.mp4")
    assert "-f" in cmd and "concat" in cmd and "-c" in cmd


def test_build_drawtext_cmd(tmp_path):
    cmd = build_drawtext_cmd(tmp_path / "i.mp4", tmp_path / "o.mp4",
                             [{"start_s": 0.0, "end_s": 2.0, "text": "知识就是力量"}],
                             tmp_path / "font.ttc", 56)
    vf = cmd[cmd.index("-vf") + 1]
    assert "drawtext=fontfile=" in vf and "知识就是力量" in vf and "between(t,0,2)" in vf
    assert "y=h*0.78" in vf
