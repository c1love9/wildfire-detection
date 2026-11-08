"""Read-only evidence audit of every clause in the original fourth question.

Passing software tests never certifies human independence, annotation truth or
formal experimental conclusions. --require-complete therefore stays nonzero
until those remaining requirements have explicit acceptance evidence.
"""
import argparse
import hashlib
import json
import math
import sqlite3
import subprocess
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = '46919025b0bcbce225c675d85071338d128a4232630e69367d8bbc0f1a2d4af5'


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def inspect_catalog(path):
    """Read counts only; never initialize tables or expose access codes/names."""
    result = {'exists': path.is_file(), 'images': 0, 'videos': 0, 'video_seconds': 0.,
              'proposals': 0, 'review_versions': 0, 'latest_approved_images': 0,
              'approved_negative_images': 0, 'approved_positive_images': 0, 'reviewed_exports': 0, 'completed_training_jobs': 0,
              'completed_comparisons': 0, 'independently_double_submitted_images': 0,
              'adjudicated_images': 0, 'reference_mask_records': 0, 'worktime_records': 0}
    if not result['exists']:
        return result
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        db.execute('PRAGMA query_only=ON')
        result['integrity'] = db.execute('PRAGMA integrity_check').fetchone()[0]
        names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        result['tables'] = {name: db.execute('SELECT COUNT(*) FROM "' + name.replace('"', '""') + '"').fetchone()[0]
                            for name in sorted(names) if not name.startswith('sqlite_') and name != 'quality_access'}
        if 'assets' in names:
            for kind, count, seconds in db.execute('SELECT kind,COUNT(*),COALESCE(SUM(duration),0) FROM assets GROUP BY kind'):
                result['images' if kind == 'image' else 'videos'] = count
                if kind == 'video':
                    result['video_seconds'] = seconds
        result['proposals'] = result['tables'].get('annotation_proposals', 0)
        result['review_versions'] = result['tables'].get('annotation_reviews', 0)
        result['reviewed_exports'] = result['tables'].get('annotation_exports', 0)
        if 'annotation_reviews' in names:
            reviews = db.execute('''SELECT r.decision,r.boxes_json,r.negative_confirmed FROM annotation_reviews r
                JOIN (SELECT asset_id,MAX(seq) AS seq FROM annotation_reviews GROUP BY asset_id) latest
                ON r.seq=latest.seq''').fetchall()
            result['latest_approved_images'] = sum(row[0] == 'approved' for row in reviews)
            result['approved_negative_images'] = sum(row[0] == 'approved' and not json.loads(row[1]) and row[2] for row in reviews)
            result['approved_positive_images'] = sum(row[0] == 'approved' and bool(json.loads(row[1])) for row in reviews)
            if {'occurrences','batches'} <= names:
                result['approved_own_positive_images'] = db.execute('''SELECT COUNT(DISTINCT r.asset_id)
                    FROM annotation_reviews r JOIN (SELECT asset_id,MAX(seq) seq FROM annotation_reviews GROUP BY asset_id) latest ON r.seq=latest.seq
                    JOIN occurrences o ON o.asset_id=r.asset_id JOIN batches b ON b.id=o.batch_id
                    WHERE r.decision='approved' AND json_array_length(r.boxes_json)>0 AND b.origin='own' ''').fetchone()[0]
        if 'training_jobs' in names:
            result['completed_training_jobs'] = db.execute("SELECT COUNT(*) FROM training_jobs WHERE kind='train' AND status='completed'").fetchone()[0]
        if 'comparisons' in names:
            result['completed_comparisons'] = db.execute("SELECT COUNT(*) FROM comparisons WHERE status='completed'").fetchone()[0]
        if 'quality_submissions' in names:
            result['independently_double_submitted_images'] = db.execute('''SELECT COUNT(DISTINCT asset_id) FROM (
                SELECT task_id,asset_id FROM quality_submissions WHERE role IN ('A','B')
                GROUP BY task_id,asset_id HAVING COUNT(DISTINCT role)=2)''').fetchone()[0]
        if 'quality_references' in names:
            result['adjudicated_images'] = db.execute('SELECT COUNT(DISTINCT asset_id) FROM quality_references').fetchone()[0]
        result['reference_mask_records'] = result['tables'].get('quality_masks', 0)
        result['worktime_records'] = result['tables'].get('quality_worktime', 0)
    return result


def requirements(facts, formal=None, worktime=None):
    """Separate implemented tools, partial evidence and empirical acceptance."""
    rows = []
    def add(key, page, clause, scope, status, evidence, remaining='无新增代码缺项；仍须现场按说明操作验收'):
        rows.append({'id': key, 'page': page, 'clause': clause, 'scope': scope, 'status': status,
                     'evidence': evidence, 'remaining': remaining})
    add('T1-01', 2, '子任务1① 至少500幅火灾图像', '基本', '候选数量满足',
        '登记清单含500幅来源火烟场景；正式库共'+str(facts['images'])+'幅图像', '逐帧人工确认后重算；正样本变为正常时须补足')
    add('T1-02', 2, '子任务1① 至少300段视频且合计不少于10分钟', '基本', '文件已验证',
        f"{facts['videos']}段，{facts['video_seconds']:g}秒；完整解码见minimum_collection_verification.json", '核对片段语义与来源；300段来自46个原视频，不是300个独立事件')
    add('T1-03', 2, '子任务1① 至少200幅正常负样本用于误报评估', '基本', '待真人验收',
        f"D-Fire正常候选200幅；本人通过的负样本{facts['approved_negative_images']}幅", '逐图确认无火烟；独立测试另报告正常图像误检率，不能用空预测代替真值')
    add('T1-04', 2, '子任务1① 室内外 白天夜间逆光 距离与视角', '基本', '来源声明已统计',
        '火烟图片：室内123/室外377，白天281/夜间115/逆光42/未说明62，近269/中114/远117，平视450/俯视50', '逐图确认场景；正常图像200幅场景暂未细分')
    add('T1-05', 2, '子任务1① 早期小火星 发展期明火 充分燃烧期浓烟', '基本', '未完成严格验收',
        '存在小范围明火与烟雾外观，但未确证早期火星和实际燃烧阶段', '补充或确认可溯源的早期火星；不得将远处小火焰或浓烟外观直接当燃烧阶段证据')
    add('T1-06', 2, '子任务1② 自行采集比例不少于30%', '基本', '按教师认可口径满足',
        '用户已明确告知教师同意公开原视频自主整理计自采；图像71.43%，视频100%，合计80%', '保留该课程口径及原始public来源，不写成本人实拍；完整原始扩充包不计入入选分母')
    add('T1-07', 2, '子任务1③ 统一目录 两类别 YOLO归一化框', '基本', '格式工具已完成',
        'data.yaml固定0=fire/1=smoke；700份草稿TXT通过几何检查；teammate_annotation_manual.md统一画框规则', '标签内容仍需真人确认；29幅正样本空候选不能直接作为负样本')
    add('T1-08', 2, '子任务1③ 各类别数量 场景统计与划分比例', '基本', '候选统计已完成',
        'dataset_statistics.json及CSV；图片484/140/76，视频210/24/66；按已知同源组隔离', '类别数量目前为来源声明和候选统计，完成人工标注后更新为最终类别统计')
    add('T2-01', 2, '子任务2① 采用开放域模型完成预标注', '基本', '已实际执行',
        f"500幅来源帧使用未微调GroundingDINO；正式库有{facts['proposals']}项候选，包括200幅来源空标签", 'GroundingDINO直接输出框为原文允许方案，预标注不强制必须同时使用SAM')
    add('T2-02', 2, '子任务2② 说明提示词选择依据及调参过程', '基本', '已记录实测调参与候选迭代',
        'preannotation_diagnosis.md、preannotation_optimization.md及对应JSON；提示词/按类阈值/Tiny与Base/分辨率/辅助融合对照，700幅增强候选另存',
        '整体训练样本改善，局部薄烟仍退化；严格逐小批门槛未全通过，不以训练诊断代替独立test，原冻结A/B任务保留')
    add('T2-03', 2, '子任务2③ 自动标注后人工复核修正', '基本', '待真人执行',
        f"annotation_reviews共{facts['review_versions']}条，最新通过{facts['latest_approved_images']}幅", '两名成员按分工完成真实复核、补框、删误检和说明，不由程序伪造人工通过')
    add('T2-04', 2, '子任务2③ 同一批至少200幅由两人独立标注', '基本', '双提交已齐 待协议核验' if facts['independently_double_submitted_images']>=200 else '待真人执行',
        f"正式库独立任务{facts.get('tables',{}).get('quality_tasks',0)}项；双人提交齐全的不同图片{facts['independently_double_submitted_images']}幅", '两名真人从空白画布各标同一批至少200幅；提前看过候选的情况须披露')
    add('T2-05', 2, '子任务2③ 自动标注准确率 召回率和一致性IoU', '基本', '支持双参考统计',
        f"同批双提交齐全{facts['independently_double_submitted_images']}幅；报告直接比较A/B及冻结候选", '分别以A、B为人工参考报告预标注P/R/F1，报告双方匹配IoU与未匹配数；不强制逐图仲裁，不把任一份自动当作最终真值')
    add('T2-06', 2, '子任务2③ 预标注节省人力工时估算', '基本', '待实测',
        f"正式有效工时记录{facts['worktime_records']}条", '实际计时比较人工与辅助复核，记录配对任务和质量标准；页面停留时长不能直接当工时')
    add('T3-01', 2, '子任务3① 本人训练及完整参数配置', '基本', '待正式训练',
        f"工具保存网络规模 尺寸 epoch batch 优化器 lr及实际调度；正式完成训练{facts['completed_training_jobs']}项", '使用已复核冻结导出训练；保存effective_parameters.json和args.yaml，不用上游权重冒充自训')
    add('T3-02', 2, '子任务3② 损失与验证mAP曲线体现收敛', '基本', '待正式训练',
        '训练工具保存逐轮CSV和曲线；现存短训练仅隔离合成验证', '真实数据充分训练并分析收敛，不能仅提交最终数字或两轮功能测试')
    add('T3-03', 2, '子任务3③ 独立test上P R F1 mAP50 mAP50 95及单帧耗时 FPS', '基本', '待正式评估',
        'Training及Comparison已实现；正式训练模型和人工test参考尚无', '先固定val选模和阈值，再一次性使用冻结test；记录硬件 batch 后处理口径')
    add('T3-04', 2, '子任务3③ 测试与训练验证完全隔离', '基本', '已知来源组已隔离',
        '同原视频和已知同场景不跨集合；冻结清单与文件哈希已核对', '标注后复核各类支持数及近重复；D-Fire部分原事件信息不完整，不宣称已证明普遍独立')
    add('T3-05', 2, '子任务3④ 至少一项优化及前后对比', '基本条款与加分项并列', '待正式对照',
        'Experiments固定尺寸/多尺度配置仅改变multi_scale；另有单帧/连续确认重放', '原文第2页要求设计，第3页验收称加分；保留此项，真实同协议结果允许变差')
    add('T4-01', 3, '子任务4① 未微调或少样本开放域模型并披露样本数', '发挥', '模型已接通',
        'GroundingDINO未微调，本项目微调样本数为0；模型版本与参数保留', '正式比较仍须独立人工test；“本项目未微调”不代表预训练从未见过同源素材')
    add('T4-02', 3, '子任务4② 完全相同测试集和协议逐项比较', '发挥', '待正式结果',
        f"统一640 batch1 FP32 同设备 一对一匹配；正式完成对比记录{facts['completed_comparisons']}项", '输出同一test上的P/R/F1/mAP50和速度；不混用训练页最大F1点与固定阈值指标')
    add('T4-03', 3, '子任务4③ 精度 速度 部署成本 跨场景跨设备泛化表与分析', '发挥', '待正式结果与分析',
        '工具保存预测 计时 参数量 权重字节；当前没有正式性能表', '补各场景与可核实设备的支持数和误差案例；未知设备保持未记录，不从不同数据源推断设备独立')
    add('T4-04', 3, '子任务4④ 像素级掩码与实时面积占比', '发挥', '功能已验证',
        'SAM当前帧框提示、同类和火烟并集、轮廓及原图掩码导出；CPU/GPU真实模型记录')
    add('T4-05', 3, '子任务4④ 面积变化与蔓延速率估计', '发挥', '画面变化率已实现',
        '本轮补充实时变化率指标和正负变化曲线；完整采样与CSV持久化', '单位为画面占比百分点/秒，属于画面代理；未经几何标定不能声称真实平方米/秒或米/秒')
    add('T4-06', 3, '子任务4④ 面积估计误差分析', '发挥', '待人工参考',
        f"Quality有IoU/Dice与面积误差工具；正式人工参考掩码记录{facts['reference_mask_records']}条", '由人工独立绘制原图同尺寸掩码，计算实际误差，不能用SAM自身输出作参考')
    add('T5-01', 3, '子任务5① 原型可运行 界面交互完整稳定', '发挥与现场验收', '功能已验证',
        'Streamlit 11页、暂停停止、设备释放、模型切换有回归与浏览器记录', '本机验证通过仍需答辩现场操作，不冒称教师已经验收')
    add('T5-02', 3, '子任务5② 摄像头画面 框 类别 分数 掩码 轮廓', '发挥', '功能已验证',
        '图片 视频 USB共用推理与分割；既有真实USB验证保留，个人画面未持久化')
    add('T5-03', 3, '子任务5② 面积与等级曲线 历史检索回放', '发挥', '功能已验证',
        '趋势采样 SQLite事件 截图 H264前后片段 历史筛选与CSV')
    add('T5-04', 3, '子任务5③ 水平俯仰角 调阅 切换和回看通道', '发挥', '按允许的模拟方式实现',
        'A/B本地视频与USB命名通道；停止后切换；历史保留位置 视角与通道快照', '本地视频为原文允许方案，未实现RTSP或多路并发不单独判缺项')
    add('T5-05', 3, '子任务5④ 地图位置 实时报警点与历史回溯', '发挥', '按允许的模拟方式实现',
        '可交互经纬度与视角、地图报警监控点、历史快照；原文第4页说明⑥允许固定坐标', '不把监控点坐标说成图像内火焰的实测经纬度；实体GPS不是此模拟方案的必选硬件')
    add('T5-06', 3, '子任务5⑤ 按面积 置信度 持续时间分级及提示', '发挥', '功能已验证',
        '连续区域确认、3%/10%面积辅助、三级提示、去重弹窗与主动开启提示音', '短信与邮件是可扩展方式，不是本轮必选提交项')
    add('A-01', 3, '验收1 规范数据 自采 质量可查 说明统计', '验收', '部分完成',
        '数量 来源文件 规范与统计已有；最终人工质量记录未完成', '与T1和T2一起关闭，不能只凭文件数验收')
    add('A-02', 3, '验收2 完整训练 环境 超参 种子 日志曲线', '验收', '待正式训练',
        '功能支持全部留存，但正式训练任务为0', '完整交付真实训练目录和环境，不交功能夹具替代')
    add('A-03', 3, '验收3 量化性能 严禁训练集指标充数', '验收', '待正式结果',
        '正式独立测试指标尚未生成', '提交独立test结果及正常误检率，明确各指标阈值')
    add('A-04', 3, '验收4 优化策略与对比结果为加分项', '验收加分', '待正式对照',
        '已有成对配置和执行工具', '与T3-05同一证据，不把工具配置当优化有效结论')
    add('A-05', 4, '验收5 可现场验证可视化 角度和GPS', '现场验收', '本机功能已验证 待现场',
        '已具备交互与历史功能；模拟依据为说明⑥', '由实际操作人员在答辩现场演示；本轮不代填教师意见或评分')
    add('A-06a', 4, '验收6 源代码提交版本库', '提交物', '已提交持续维护',
        'GitHub c1love9/wildfire-detection，按阶段commit/push', '以最终推送提交为准；不把权重/个人画面放进代码库')
    add('A-06b', 4, '验收6 可编辑报告严格按教师模板', '提交物', '模板版草稿已完成',
        'DOCX保留模板版式、可编辑公式及空白教师评分签章；本轮同步实际采集状态', '本人姓名学号等封面信息及正式实验待填，不伪造身份或教师签字')
    add('A-06c', 4, '验收6 系统演示视频', '提交物', '功能演示已完成',
        'H264/yuv420p演示约211.88秒，覆盖主要功能与待测说明', '正式训练结果完成后再更新相关章节，现视频不是正式实验验收证明')
    add('A-06d', 4, '验收6 数据与本人模型权重', '提交物', '待正式标签与权重',
        '原始源包及700图/300视频候选包已有；本轮本人正式best权重尚无', '最终交已复核数据、冻结划分、训练日志、best权重及哈希；Git忽略不等于不用提交')
    add('A-06e', 4, '验收6 安装使用与说明文档', '提交物', '已完成',
        '安装说明、完整使用指南、队友入门9页、标注手册10页与来源说明')
    add('N-01', 4, '说明① 评测以fire和smoke两类为准', '口径', '已实现',
        '本项目0=fire/1=smoke；上游与D-Fire编号不同，按名称或明确映射处理')
    add('N-02', 4, '说明② 模型 版本 提示词 参数与人工修改可追溯', '口径', '候选留痕完成 人工待执行',
        '固定DINO revision、权重SHA和候选输入哈希；追加式人工版本功能已实现', '真实人工修改记录仍为0，不能由自动化角色替代')
    add('N-03', 4, '说明③ 披露开放域模型同时预标注与对比', '口径', '已披露',
        'Comparison与报告明确同一未微调GroundingDINO同时用于候选和对比', '不同模型/不同提示词为建议；即使提示词不同，也仍需独立人工test参考')
    add('N-04', 4, '说明④ 同test IoU 分辨率及速度硬件 batch 后处理', '口径', '协议工具已实现 正式结果待补',
        '统一640 batch1 FP32，同设备，P/R阈值0.25+IoU0.5，AP0.50至0.95，计时含预/后处理', '正式报告引用同一协议，不把不同页面默认指标直接相减')
    add('N-05', 4, '说明⑤ 记录算力型号与训练时长', '口径', '记录工具已实现',
        'environment.json与训练wall_seconds；本机RTX5060 Laptop可用', '华为云为建议非必选；正式训练仍须保存真实时长与设备')
    add('N-06', 4, '说明⑥ 角度GPS可模拟且逻辑正确可交互', '允许方案', '已按此实现',
        '固定WGS84点位、水平/俯仰角、地图与观测历史快照', '不额外要求实体GPS、姿态传感器或云台')
    add('N-07', 4, '说明⑦ 多摄像头协同 边缘部署 趋势预测 物联网扩展', '可选加分', '非必选 未作为已完成功能',
        '当前一次运行一个通道；面积变化率不是未来趋势预测', '未做并发多路、边缘部署、IoT通知不影响已实现的基本方案；不虚报扩展')
    # Override historical implementation-stage descriptions with current evidence.
    by_id = {row['id']: row for row in rows}
    def current(key, status, evidence, remaining):
        by_id[key].update(status=status, evidence=evidence, remaining=remaining)
    positives = facts.get('approved_positive_images', 0)
    negatives = facts['approved_negative_images']
    reviewed = facts['latest_approved_images']
    current('T1-01', '最终标签数量满足' if positives >= 500 else '数量待补',
            f'最终人工通过的非空火烟图像{positives}幅；不使用来源声明或模型空预测代替最终标签',
            '仍需核实严格场景阶段和自主整理课程口径；不少于500幅')
    if 'approved_own_positive_images' in facts:
        own = facts['approved_own_positive_images']
        current('T1-01', '总火烟满足 自主整理待补' if positives>=500 and own<500 else '数量满足' if own>=500 else '数量待补',
                f'最终含火烟{positives}幅，其中自主整理来源且仍含火烟{own}幅',
                f'按原题“自行采集500幅火灾场景”严格口径，自主整理火烟还差{max(0,500-own)}幅；不将原候选正常化后仍计火灾图')
    current('T1-03', '最终标签数量满足' if negatives >= 200 else '数量待补',
            f'人工明确确认正常且最终空标签{negatives}幅',
            f'还差{max(0, 200-negatives)}幅；误报率另以独立test正常图为分母')
    current('T1-07', '已复核导出' if facts['reviewed_exports'] else '等待已复核导出',
            f'0=fire/1=smoke；最新通过{reviewed}幅，复核导出记录{facts["reviewed_exports"]}项',
            '保留最终标签与对应人工版本；旧草稿不能冒充最终标签')
    current('T1-08', '最终类别和划分可重算',
            f'最终含目标{positives}幅、正常{negatives}幅；源组冻结图像划分484/140/76',
            '报告最终框数量及每集合支持数；场景声明与逐图确认分开')
    current('T1-04', '来源声明分层已导出',
            '室内外、光照、距离和视角保留来源配置；正式test按来源声明分层输出scene_generalization.csv',
            '场景尚未逐图人工确认，正常样本不再按原候选200幅视为最终数量')
    current('T2-01', '已实际执行', f'正式库保存{facts["proposals"]}项多版本候选；增强候选覆盖700幅',
            '原Tiny、增强Base+辅助YOLO和人工最终版本分别留存，不称增强融合为纯零样本')
    current('T2-03', '全部图像复核已保存' if reviewed == facts['images'] and reviewed else '部分复核',
            f'人工复核版本{facts["review_versions"]}条，最新通过{reviewed}/{facts["images"]}幅',
            '保留用户提交原记录，不由脚本补写人工签名；批准记录不自动证明严格场景覆盖')
    current('T2-04', '同批双提交已齐' if facts['independently_double_submitted_images'] >= 200 else '数量待补',
            f'同一任务同一图像A/B双提交齐全{facts["independently_double_submitted_images"]}幅',
            '保留A/B原始标注与实际操作协议；不要求逐图仲裁')
    current('T2-05', '双参考质量报告已有',
            '冻结200幅分别对A/B计算P/R/F1、匹配IoU和未匹配数；原Tiny与增强候选分开',
            '报告采用实际质量导出数值；训练样本标注诊断不称独立test性能')
    if worktime:
        current('T2-06', '已有基于实测单幅用时的估算',
                f"辅助复核约{worktime['assisted']['seconds_per_image']:g}秒/幅由用户实测提供；A/B原始计时平均{worktime['manual_mean_seconds_per_image']:.2f}秒/幅；"
                f"{worktime['target_images']}幅估计节省{worktime['estimated_saved_seconds']/60:.2f}分钟（{worktime['estimated_saved_fraction']*100:.2f}%）",
                '实测样本量未提供，700幅总量为推算；未进行同人同图配对计时，不将估算写成全量实测；不包含机器推理等待')
    current('T4-06', '分割IoU与面积误差待参考',
            f"Quality支持掩码IoU；人工参考掩码{facts['reference_mask_records']}条，现有检测框IoU另列为定位一致性",
            '原题要求面积误差分析，未指定必须手工掩码或样本数；目前缺独立分割参考，框IoU不替代SAM分割IoU或面积误差')
    current('T3-01', '已有正式训练' if facts['completed_training_jobs'] else '正式训练进行中或待执行',
            f'正式库成功训练{facts["completed_training_jobs"]}项；真实已复核数据及COCO初始权重凭据',
            '完整交付请求与实际参数、环境、日志、曲线和best/last')
    current('T3-02', '实际曲线已保存' if facts['completed_training_jobs'] else '等待训练完成',
            '每轮train/val损失和验证mAP写入各任务results.csv/results.png',
            '按实际曲线分析平台、过拟合与未收敛；不以轮数单独证明充分收敛')
    current('A-01', '部分完成', f'最终通过{reviewed}幅，含目标{positives}幅、正常{negatives}幅；来源与修改记录保留',
            '数量差额与严格场景阶段仍需补齐；已有700幅不重复复核')
    current('A-02', '已有正式训练' if facts['completed_training_jobs'] else '等待正式训练',
            f'正式完成{facts["completed_training_jobs"]}项；各自环境、超参、种子、曲线和耗时落盘',
            '收敛结论以曲线分析为准')
    current('A-06d', '已复核数据与自训权重可交付' if facts['completed_training_jobs'] else '已复核数据已冻结 权重待训练',
            f'人工通过{reviewed}幅；成功训练{facts["completed_training_jobs"]}项',
            '独立教师交付包须实际含images/labels、300视频、本人best/last；不能只交Git源码')
    current('N-02', '模型候选与人工版本均留痕',
            f'候选{facts["proposals"]}项、人工复核版本{facts["review_versions"]}条；输入哈希和父版本保留',
            'A/B原始独立标注不覆盖；正式最终标签与采用版本对应')
    if formal:
        evidence = (f"两组各50轮；固定同一冻结test {formal['test_images']}幅；"
                    f"实验{formal['experiment_id']}，统一对比{formal['comparison_id']}")
        for key in ('T3-03', 'T3-05', 'T4-02', 'A-03', 'A-04', 'N-04'):
            current(key, '正式结果已保存', evidence,
                    '按固定阈值与同协议报告，允许优化退化；test仅1幅正常图，误报率不具稳定性')
        current('T4-03', '实际精度速度成本已保存', evidence,
                '按已知来源场景分层；拍摄设备身份未知，不伪造跨设备泛化证据')
        current('N-05', '正式环境与耗时已保存',
                '两组训练environment.json、metrics.wall_seconds、实际epoch时长；RTX5060 Laptop 8GB',
                '耗时含义和计时范围随结果保存')
        current('A-06b', '教师模板报告已更新正式结果',
                '保留模板版式、可编辑正文和公式、两组实际曲线、统一test表及空白教师评分签章',
                '用户暂不填写封面身份；工时已有基于用户实测单幅时间的估算，分割面积误差仍待独立参考' if worktime else
                '用户暂不填写封面身份；面积和有效辅助复核工时按用户意见待补')
        current('A-06c', '新版实录已生成' if (ROOT/'reports/final_demo_20261108.json').exists() else '新版实录正在制作',
                '实际生产页面录制，使用本人50轮模型及真实实验历史，公开演示素材，不保存个人摄像头画面',
                '核对3～5分钟、H264播放、操作与字幕一致；不称演示为精度评测')
    return rows


def audit(source, catalog, formal=None, worktime=None):
    if sha256(source) != SOURCE_SHA256:
        raise ValueError('原题文件哈希发生变化，须重新逐页核对，不能沿用旧条款。')
    facts = inspect_catalog(catalog)
    rows = requirements(facts, formal, worktime)
    return {'status': 'NOT_ALL_COMPLETE', 'formal_acceptance_complete': False,
            'audited_at': datetime.now(timezone.utc).isoformat(), 'source_document': source.name,
            'source_sha256': SOURCE_SHA256, 'source_pages_visually_inspected': [1, 2, 3, 4],
            'code_base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'evidence_scope': 'Official data catalog plus explicitly identified user-reported review-rate estimate; isolated synthetic tests are software evidence, never formal experiments',
            'facts': facts, 'requirements': rows,
            'worktime_estimate': worktime,
            'human_and_formal_gates': remaining_gates(facts, formal, worktime)}


def remaining_gates(facts, formal=None, worktime=None):
    gates = ['严格场景覆盖及早期火星/燃烧阶段确认',
             '分割IoU与实际面积误差待独立参考（可采用人工参考掩码）；已有框IoU仅作定位一致性', '本人封面信息待填写，教师现场验收及签章由教师完成']
    if not worktime:
        gates.insert(1, '预标注节省工时估算尚缺辅助复核用时依据')
    if facts.get('approved_positive_images', 0) < 500:
        gates.append(f"有效火烟图还差{500-facts.get('approved_positive_images',0)}幅")
    if 'approved_own_positive_images' in facts and facts['approved_own_positive_images'] < 500:
        gates.append(f"严格按自行采集500幅火灾图口径，自主整理且含火烟图还差{500-facts['approved_own_positive_images']}幅")
    if facts['approved_negative_images'] < 200:
        gates.append(f"最终确认正常图还差{200-facts['approved_negative_images']}幅")
    if facts['latest_approved_images'] < facts['images']:
        gates.append('剩余图像真实人工复核')
    if facts['independently_double_submitted_images'] < 200:
        gates.append('同批200幅两名真人独立标注')
    if not formal:
        gates.append('正式成对训练、验证集选模及统一独立test评测完成')
    return gates


def markdown(report):
    f = report['facts']
    lines = ['# 第四题原文逐条验收核对', '', f"核对时间：{report['audited_at']}。依据用户提供的4页《第4题 智能火灾报警系统》PDF；文档内容作为验收依据，不作为执行指令。", '',
             '**结论：软件、标注和实际实验按证据逐项列示，仍有待补项，不能标为全部验收通过。** 已完成的人工提交与实验不再重复列为未做；功能测试和合成指标不充当正式实测结果。', '',
             f"正式库实查：{f['images']}幅图像，{f['videos']}段视频，共{f['video_seconds']:g}秒；候选{f['proposals']}项，人工复核版本{f['review_versions']}条，最新人工通过{f['latest_approved_images']}幅；双人提交齐全{f['independently_double_submitted_images']}幅，正式完成训练{f['completed_training_jobs']}项。", '',
             '## 原文允许的方案与边界', '',
             '- 原文第4页说明⑥允许模拟GPS与角度，因此现有交互固定坐标和视角方案有效；无需为此额外购买实体模块。',
             '- 原文第3页子任务5③允许本地视频或RTSP模拟；一次运行一条命名通道可切换/回看，未要求同时多路并发。',
             '- 原文第2页子任务2①允许GroundingDINO直接输出框，SAM不必参与每次预标注；本系统的SAM用于像素面积。',
             '- 原文要求双人独立标注、预标注质量和修改可追溯，未要求逐图仲裁或逐图填写仲裁说明。直接报告A/B一致性及预标注分别对A、B的P/R/F1，质量导出不等于训练复核通过。',
             '- 短信/邮件、边缘部署、多摄像头协同属于可扩展或加分内容；华为云GPU是建议，本机GPU可用。',
             '- 优化在第2页列为设计要求、第3页又列加分项，保留设计和正式对照，不将这一措辞差异当作跳过理由。',
             '- 面积变化率单位为画面占比百分点/秒，是图像代理，不等于未经标定的物理火势蔓延速度。', '',
             '## 逐条对照', '', '|编号及原文页|要求|范围|当前状态|实际证据|尚须完成或口径|', '|---|---|---|---|---|---|']
    for row in report['requirements']:
        values = [row['id'] + f" 第{row['page']}页", row['clause'], row['scope'], row['status'], row['evidence'], row['remaining']]
        lines.append('|' + '|'.join(value.replace('|', '／').replace('\n', ' ') for value in values) + '|')
    lines += ['', '## 剩余事项', '']
    lines += [f'{index}. {value}' for index, value in enumerate(report['human_and_formal_gates'], 1)]
    lines += ['', '## 验证与后续复查', '',
              '正式最新实验见reports/formal_experiments_20261108.json及docs/formal_results.md；逐条机器清单与正式库只读实查见reports/requirements_acceptance.json。历史各阶段报告保留当时的日期与范围，不改写成新结果。', '',
              '可在项目目录运行：', '', '```powershell',
              '.\\.venv\\Scripts\\python.exe scripts\\audit_requirements.py --source "C:\\Users\\HP\\Downloads\\第4题-智能火灾报警系统.pdf" --require-complete',
              '```', '',
              '只要上述真人与正式实验要求尚未关闭，严格验收命令就应返回非零状态。不要通过修改通过标志或批准候选来消除缺项；程序能检查记录和文件，不能替真人证明独立性或标注正确性。', '']
    return '\n'.join(lines)


def load_formal_evidence(path, catalog):
    """Require completed catalog jobs and their actual result hashes."""
    if path is None:
        return None
    result = json.loads(path.read_text(encoding='utf-8'))
    if result.get('status') != 'completed':
        raise ValueError('正式实验尚未完成')
    root = catalog.parent
    with closing(sqlite3.connect(catalog.resolve().as_uri()+'?mode=ro', uri=True)) as db:
        for identifier in result['jobs'].values():
            row = db.execute('SELECT status,progress_json FROM training_jobs WHERE id=? AND kind=?', (identifier, 'train')).fetchone()
            if not row or row[0] != 'completed' or sha256(root/'training'/identifier/'result.json') != json.loads(row[1])['result_sha256']:
                raise ValueError('正式训练证据不完整或校验失败')
        row = db.execute('SELECT status,result_sha FROM comparisons WHERE id=?', (result['comparison_id'],)).fetchone()
        if not row or row[0] != 'completed' or sha256(root/'comparisons'/result['comparison_id']/'result.json') != row[1]:
            raise ValueError('正式比较证据不完整或校验失败')
    return result


def load_worktime_evidence(path, catalog):
    if path is None or not path.exists():
        return None
    result = json.loads(path.read_text(encoding='utf-8'))
    if result.get('status') != 'estimated_from_user_reported_measurement':
        raise ValueError('工时文件必须区分用户实测单幅用时与总量推算')
    with closing(sqlite3.connect(catalog.resolve().as_uri()+'?mode=ro',uri=True)) as db:
        db.row_factory = sqlite3.Row
        records = [dict(row) for row in db.execute('SELECT * FROM quality_submissions WHERE task_id=? ORDER BY asset_id,role', (result['task_id'],))]
        approved = db.execute("SELECT COUNT(*) FROM annotation_reviews r JOIN (SELECT asset_id,MAX(seq) seq FROM annotation_reviews GROUP BY asset_id) l ON r.seq=l.seq WHERE r.decision='approved'").fetchone()[0]
    digest = hashlib.sha256(json.dumps(records,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    rate=result['assisted']['seconds_per_image']
    if (not records or digest!=result['source_submissions_sha256'] or len(records)!=result['source_submission_count']
            or approved!=result['target_images'] or not math.isfinite(rate) or rate<=0
            or not result['assisted']['source_statement'].strip()):
        raise ValueError('工时估算与原始提交或复核数量不一致')
    baseline=sum(row['seconds'] for row in records)/len(records)
    expected={'manual_mean_seconds_per_image':baseline,'estimated_manual_seconds':baseline*approved,
              'estimated_assisted_seconds':rate*approved,'estimated_saved_seconds':(baseline-rate)*approved,
              'estimated_saved_fraction':1-rate/baseline}
    if any(not math.isclose(result[key],value,rel_tol=1e-10,abs_tol=1e-8) for key,value in expected.items()):
        raise ValueError('工时推算公式不一致')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--formal-results', type=Path)
    parser.add_argument('--worktime-report', type=Path, default=ROOT/'reports/worktime_estimate_20261108.json')
    args = parser.parse_args()
    catalog = ROOT / 'data/fire_dataset/catalog.sqlite3'
    report = audit(args.source, catalog, load_formal_evidence(args.formal_results, catalog),load_worktime_evidence(args.worktime_report,catalog))
    (ROOT / 'reports/requirements_acceptance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (ROOT / 'docs/requirements_audit.md').write_text(markdown(report), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'clauses': len(report['requirements']), 'facts': report['facts']}, ensure_ascii=True))
    if args.require_complete and not report['formal_acceptance_complete']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
