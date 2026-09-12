# DECISIONS（只记影响未来开发的，2026-09-13 建）

## D1 身份判等只认真值三源
Context: 外观别名子串匹配把不同人合成同一 canonical → det 假绿（V3_C3 badcase）。
Alternatives: 别名降权 / 删别名。
Chosen: registry 四键——只有 aliases(身份名)/source_entities(窗口作用域)/
supported+verified bindings 进硬判等；appearance/role 只做绑定引导。
Consequences: 漏判↑（少连续加分），换来假绿消失；解锁靠 bindings 产出。

## D2 3 窗门控是全量重标的启动门
Context: pack-on 标注是新高风险环节（可能引入确认偏倚/幻觉绑定）。
Chosen: 指定窗小测试（同人多呈现 + hard negative）五判据全过才放 72 窗。
Consequences: 五轮抓出 9 个会污染全量的 bug；此模式沿用到三臂实验。

## D3 审谁交谁 + 诚实失败
Context: 盲看看的是素材原声版，交付的却是纯 BGM 版；失败仍标记 complete。
Chosen: 盲看挪到终版之后；overall 门控失败=delivery blocked（仍出调试预览）。
Consequences: V4_C 出现合法黑屏+blocked——正确行为，解锁在标注层。

## D4 对白承载内容的素材保留原声
Context: 验证靠对白通过、交付却删对白（换纯 BGM+裁字幕）。
Chosen: audio 三态；mix=原声 1.0 + BGM 0.25（amix normalize=0 显式）。
Consequences: 0.25 是默认非标准，验收以人耳听清为准。

## D5 分片并行是批处理默认形态
Context: 单实例串行 72 窗要 5h+。
Chosen: output_path 分片输出 + merge_annotation_shards 并集合并入库。
Consequences: 4 实例八卡 ~1.2h；ops 铁律：setsid 脱离/flock 单实例/
watchdog 有界重拉/pgrep 锚定完整路径/Windows 脚本上传前 sed 去 CRLF。
