"""
RoseGrade 自适应鲜切花卉智能分级系统 - 主入口

用法:
    python main.py collect          # 收集基准资产
    python main.py detect           # 执行漂移检测
    python main.py grade <image>    # 分级单张图片
    python main.py serve            # 启动 REST API 服务
    python main.py demo             # 运行动态检测演示流水线
    python main.py visualize        # 生成可视化报告
"""
import argparse
import sys
import os


def cmd_collect(args):
    """收集基准资产"""
    from src.baseline_collector import YOLO11AutoCollector
    from src.utils import BASELINE_ASSETS_DIR
    
    # 使用默认参数或 args 中的参数
    model_path = args.model or os.path.join('runs', 'classify', 'train2', 'weights', 'best.pt')
    dataset_root = args.dataset or 'dataset'
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        sys.exit(1)
    
    if not os.path.exists(dataset_root):
        print(f"错误: 数据集目录不存在: {dataset_root}")
        sys.exit(1)
    
    print(f"开始收集基准资产...")
    print(f"   模型路径: {model_path}")
    print(f"   数据集目录: {dataset_root}")
    
    try:
        collector = YOLO11AutoCollector(
            model_path=model_path,
            dataset_root=dataset_root
        )
        df = collector.run()
        if df is not None:
            collector.save_assets(df)
            print("基准资产收集完成！")
            print(f"   资产保存位置: {BASELINE_ASSETS_DIR}")
        else:
            print("未收集到有效数据")
            sys.exit(1)
    except Exception as e:
        print(f"收集基准资产失败: {e}")
        sys.exit(1)


def cmd_detect(args):
    """执行漂移检测"""
    from src.drift_report import DriftReportGenerator
    from src.utils import BASELINE_ASSETS_DIR
    
    output = args.output or 'drift_report.json'
    
    # 检查必要的基准文件是否存在
    baseline_path = os.path.join(BASELINE_ASSETS_DIR, "baseline_db.pkl")
    test_path = os.path.join(BASELINE_ASSETS_DIR, "val_test_data.pkl")
    
    if not os.path.exists(baseline_path):
        print(f"错误: 基准数据不存在: {baseline_path}")
        print("   请先运行 'python main.py collect' 收集基准资产")
        sys.exit(1)
    
    if not os.path.exists(test_path):
        print(f"错误: 测试数据不存在: {test_path}")
        print("   请先运行 'python main.py collect' 收集基准资产")
        sys.exit(1)
    
    print(f"开始执行漂移检测...")
    print(f"   基准数据: {baseline_path}")
    print(f"   测试数据: {test_path}")
    
    try:
        generator = DriftReportGenerator(baseline_path, test_path)
        report = generator.generate_report(output_path=output)
        
        # 打印简要结果
        is_drift = report.get('decision', {}).get('is_drift', False)
        mmd_score = report.get('statistics', {}).get('mmd_score', 0)
        status = report.get('decision', {}).get('status', 'UNKNOWN')
        
        print("\n" + "=" * 50)
        if is_drift:
            print("检测到数据漂移！")
        else:
            print("数据分布稳定")
        print(f"   MMD 分数: {mmd_score:.4f}")
        print(f"   状态: {status}")
        print("=" * 50)
        print(f"\n漂移检测完成！报告已保存至: {output}")
    except Exception as e:
        print(f"漂移检测失败: {e}")
        sys.exit(1)


def cmd_grade(args):
    """分级单张/批量图片"""
    from src.quality_grader import QualityGrader
    
    season = args.season or 'spring'
    
    try:
        grader = QualityGrader(season=season)
        
        if args.image:
            # 单张图片分级
            if not os.path.exists(args.image):
                print(f"错误: 图片文件不存在: {args.image}")
                sys.exit(1)
            
            print(f"开始分级单张图片: {args.image}")
            print(f"   季节参数: {season}")
            
            result = grader.grade_single(image_path=args.image)
            
            # 漂亮地格式化输出结果
            print("\n" + "=" * 60)
            print("分级结果")
            print("=" * 60)
            print(f"图片路径: {args.image}")
            print(f"最终等级: {result['grade']} 级")
            print(f"融合置信度: {result['confidence']:.4f}")
            print(f"融合分数: {result['fusion_score']:.4f}")
            print("-" * 60)
            print("YOLO 模型结果:")
            yolo = result['yolo_result']
            print(f"  预测等级: {yolo['grade']} 级")
            print(f"  置信度: {yolo['confidence']:.4f}")
            print(f"  Top-5 预测: {yolo['top5']}")
            print("-" * 60)
            print("HSV 颜色分析结果:")
            hsv = result['hsv_result']
            print(f"  颜色等级: {hsv['grade']} 级")
            print(f"  置信度: {hsv['confidence']:.4f}")
            print(f"  主色调: {hsv['features']['dominant_hue_name']}")
            print(f"  饱和度评分: {hsv['features']['saturation_score']:.4f}")
            print(f"  亮度评分: {hsv['features']['brightness_score']:.4f}")
            print(f"  色调集中度: {hsv['features']['hue_concentration']:.4f}")
            print("-" * 60)
            print(f"融合权重: YOLO={result['fusion_weights']['yolo']}, HSV={result['fusion_weights']['hsv']}")
            print(f"处理时间: {result['timestamp']}")
            print("=" * 60)
            
        elif args.dir:
            # 批量分级
            if not os.path.exists(args.dir):
                print(f"错误: 目录不存在: {args.dir}")
                sys.exit(1)
            
            print(f"开始批量分级目录: {args.dir}")
            print(f"   季节参数: {season}")
            
            results = grader.grade_batch(image_dir=args.dir)
            
            # 统计结果
            success_count = len([r for r in results if 'error' not in r])
            total_count = len(results)
            
            # 等级分布统计
            grade_counts = {1: 0, 2: 0, 3: 0, 4: 0}
            for r in results:
                if 'error' not in r:
                    grade_counts[r['grade']] += 1
            
            print("\n" + "=" * 60)
            print("批量分级统计")
            print("=" * 60)
            print(f"总计处理: {total_count} 张图片")
            print(f"成功: {success_count} 张")
            print(f"失败: {total_count - success_count} 张")
            print("-" * 60)
            print("等级分布:")
            for grade in range(1, 5):
                count = grade_counts[grade]
                percentage = (count / success_count * 100) if success_count > 0 else 0
                bar = "█" * int(percentage / 5)
                print(f"  等级 {grade}: {count:3d} 张 ({percentage:5.1f}%) {bar}")
            print("=" * 60)
            
            # 显示前5个结果的详细信息
            print("\n前 5 张图片的分级详情:")
            for i, r in enumerate(results[:5]):
                if 'error' not in r:
                    print(f"  {i+1}. {r['image_name']}: 等级 {r['grade']} (置信度: {r['confidence']:.4f})")
                else:
                    print(f"  {i+1}. {r['image_name']}: 处理失败 - {r['error']}")
            
        else:
            print("错误: 请指定 --image 或 --dir 参数")
            sys.exit(1)
            
    except Exception as e:
        print(f"分级失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def cmd_serve(args):
    """启动 REST API 服务"""
    from src.quality_grader import start_server
    
    host = args.host or '0.0.0.0'
    port = args.port or 8080
    
    print("=" * 60)
    print("RoseGrade 花卉智能分级 API 服务")
    print("=" * 60)
    print(f"服务地址: http://{host}:{port}")
    print(f"API 文档: http://{host}:{port}/docs")
    print(f"健康检查: http://{host}:{port}/api/health")
    print("-" * 60)
    print("可用 API 端点:")
    print("  POST /api/grade_single    - 单张图片分级")
    print("  POST /api/grade_batch     - 批量图片分级")
    print("  GET  /api/drift_status    - 获取漂移检测状态")
    print("  GET  /api/drift_report    - 获取完整漂移报告")
    print("  POST /api/trigger_detection - 触发漂移检测")
    print("  GET  /api/model_info      - 获取模型信息")
    print("=" * 60)
    print("按 Ctrl+C 停止服务\n")
    
    try:
        start_server(host=host, port=port)
    except KeyboardInterrupt:
        print("\n\n服务已停止")
    except Exception as e:
        print(f"\n服务启动失败: {e}")
        sys.exit(1)


def cmd_demo(args):
    """运行动态检测演示流水线"""
    from src.dynamic_detection_pipeline import DynamicDetectionPipeline
    
    n_windows = args.windows or 4
    perturbation_window = args.perturbation_window or 2
    window_size = args.window_size or 40
    drift_threshold = args.drift_threshold or 0.05
    
    print("=" * 60)
    print("RoseGrade 动态检测演示流水线")
    print("=" * 60)
    print(f"配置参数:")
    print(f"  窗口数量: {n_windows}")
    print(f"  扰动窗口索引: {perturbation_window} (第{perturbation_window + 1}个窗口)")
    print(f"  每窗口样本数: {window_size}")
    print(f"  漂移阈值: {drift_threshold}")
    print("=" * 60 + "\n")
    
    try:
        pipeline = DynamicDetectionPipeline()
        result = pipeline.run_pipeline(
            n_windows=n_windows,
            perturbation_window=perturbation_window,
            window_size=window_size,
            drift_threshold=drift_threshold
        )
        
        # 生成汇总报告
        report_dir = pipeline.generate_summary_report(result)
        
        # 打印汇总
        print("\n" + "=" * 60)
        print("动态检测演示流水线完成！")
        print("=" * 60)
        
        windows = result.get('windows', [])
        if windows:
            print(f"\n窗口处理结果汇总:")
            for w in windows:
                window_id = w['window_id'] + 1
                mmd = w.get('mmd_score', 0)
                is_drift = w.get('is_drift', False)
                triggered = w.get('triggered_training', False)
                has_perturbation = w.get('has_perturbation', False)
                
                status_icon = "" if is_drift else ""
                perturbation_mark = " [HSV扰动]" if has_perturbation else ""
                training_mark = " [触发训练]" if triggered else ""
                
                print(f"  窗口 {window_id}{perturbation_mark}: MMD={mmd:.4f}, "
                      f"状态={'漂移' if is_drift else '稳定'}{training_mark}")
            
            training_result = result.get('training_result')
            if training_result:
                print(f"\n增量训练信息:")
                print(f"  触发窗口: 窗口 {training_result.get('triggered_at_window', 0) + 1}")
                print(f"  融合模型: {training_result.get('fused_model', 'N/A')}")
        
        print(f"\n报告保存位置: {report_dir}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n演示流水线执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def cmd_visualize(args):
    """生成可视化报告"""
    from src.drift_visualizer import DriftVisualizer
    
    report_path = args.report or 'drift_report.json'
    output_dir = args.output or 'reports'
    
    if not os.path.exists(report_path):
        print(f"错误: 漂移报告不存在: {report_path}")
        print("   请先运行 'python main.py detect' 生成漂移报告")
        sys.exit(1)
    
    print(f"开始生成可视化报告...")
    print(f"   输入报告: {report_path}")
    print(f"   输出目录: {output_dir}")
    
    try:
        viz = DriftVisualizer(output_dir=output_dir)
        html_path = viz.generate_full_report(drift_report_path=report_path)
        
        if html_path:
            print("\n" + "=" * 60)
            print("可视化报告生成完成！")
            print("=" * 60)
            print(f"HTML报告: {html_path}")
            print(f"图表目录: {output_dir}")
            print("\n生成的图表包括:")
            print("  - 类别漂移热力图 (class_drift_heatmap.png)")
            print("  - 特征重要性图 (feature_importance.png)")
            print("  - 样本异常分布 (anomaly_distribution.png)")
            print("=" * 60)
        else:
            print("可视化报告生成失败，请检查报告文件格式")
            sys.exit(1)
            
    except Exception as e:
        print(f"生成可视化报告失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='RoseGrade 自适应鲜切花卉智能分级系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py collect                    收集基准资产
  python main.py detect                     执行漂移检测
  python main.py grade --image flower.jpg   分级单张图片
  python main.py grade --dir dataset/val/1  批量分级
  python main.py serve --port 8080          启动API服务
  python main.py demo                       运行动态检测演示
  python main.py visualize                  生成可视化报告
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # collect 子命令
    p_collect = subparsers.add_parser('collect', help='收集基准资产')
    p_collect.add_argument('--model', help='模型路径 (默认: runs/classify/train2/weights/best.pt)')
    p_collect.add_argument('--dataset', help='数据集路径 (默认: dataset)')
    
    # detect 子命令
    p_detect = subparsers.add_parser('detect', help='执行漂移检测')
    p_detect.add_argument('--output', '-o', help='报告输出路径 (默认: drift_report.json)')
    
    # grade 子命令
    p_grade = subparsers.add_parser('grade', help='图片分级')
    p_grade.add_argument('--image', help='单张图片路径')
    p_grade.add_argument('--dir', help='批量分级目录')
    p_grade.add_argument('--season', default='spring', 
                         choices=['spring', 'summer', 'autumn', 'winter'],
                         help='季节参数 (默认: spring)')
    
    # serve 子命令
    p_serve = subparsers.add_parser('serve', help='启动REST API服务')
    p_serve.add_argument('--host', default='0.0.0.0', help='绑定地址 (默认: 0.0.0.0)')
    p_serve.add_argument('--port', type=int, default=8080, help='端口号 (默认: 8080)')
    
    # demo 子命令
    p_demo = subparsers.add_parser('demo', help='运行动态检测演示流水线')
    p_demo.add_argument('--windows', type=int, default=4, help='窗口数量 (默认: 4)')
    p_demo.add_argument('--perturbation-window', type=int, default=2, 
                        help='施加扰动的窗口索引，0-indexed (默认: 2)')
    p_demo.add_argument('--window-size', type=int, default=40, help='每窗口样本数 (默认: 40)')
    p_demo.add_argument('--drift-threshold', type=float, default=0.05, 
                        help='触发训练的MMD阈值 (默认: 0.05)')
    
    # visualize 子命令
    p_viz = subparsers.add_parser('visualize', help='生成可视化报告')
    p_viz.add_argument('--report', help='漂移报告JSON路径 (默认: drift_report.json)')
    p_viz.add_argument('--output', '-o', help='输出目录 (默认: reports)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # 路由到对应命令
    commands = {
        'collect': cmd_collect,
        'detect': cmd_detect,
        'grade': cmd_grade,
        'serve': cmd_serve,
        'demo': cmd_demo,
        'visualize': cmd_visualize,
    }
    
    commands[args.command](args)


if __name__ == '__main__':
    main()
