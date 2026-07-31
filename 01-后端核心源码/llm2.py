def generate_results_from_temp_data():
    """从已保存的临时结果生成最终统计和图表（不需要重新跑实验）"""
    config = load_llm_config()
    logger = setup_llm_logger(config)

    # 加载已保存的5400条实验数据
    temp_file = Path(config.result_dir) / "temp_results_5400.csv"
    if not temp_file.exists():
        logger.error(f"❌ 未找到临时结果文件: {temp_file}")
        logger.error("请检查llm_experiment_results目录下是否存在temp_results_5400.csv")
        return

    logger.info(f"✅ 加载已保存的实验数据: {temp_file}")
    logger.info(f"📊 总数据条数: {len(pd.read_csv(temp_file))} 条")
    raw_df = pd.read_csv(temp_file, encoding="utf-8-sig")

    # ====================== 1. 生成核心统计结果 ======================
    logger.info("\n🔄 正在生成核心统计结果...")
    stat_df = raw_df.groupby(["模型名称", "提示词方法"]).agg({
        "解析成功率": ["mean", "std", stats.sem],
        "解析正确率": ["mean", "std", stats.sem],
        "响应时间(s)": ["mean", "std", stats.sem],
        "意图识别准确率": ["mean", "std", stats.sem],
        "参数抽取准确率": ["mean", "std", stats.sem],
        "合法JSON率": ["mean", "std", stats.sem],
        "参数匹配F1": ["mean", "std", stats.sem],
        "失败兜底成功率": ["mean", "std", stats.sem]
    }).round(4)

    # 保存核心统计结果（直接用于论文表6）
    stat_df.to_csv(Path(config.result_dir) / "llm_experiment_statistics_extended.csv", encoding="utf-8-sig")
    logger.info("✅ 核心统计结果已保存至: llm_experiment_results/llm_experiment_statistics_extended.csv")

    # ====================== 2. 生成顶刊级可视化图表 ======================
    logger.info("\n🖼️  正在生成顶刊级可视化图表...")
    plot_llm_results(raw_df, stat_df, config, logger)
    logger.info("✅ 可视化图表已保存至: llm_experiment_results/llm_parsing_results_extended.png")

    # ====================== 3. 生成多维度统计结果 ======================
    logger.info("\n📈 正在生成多维度统计结果...")

    # 3.1 不同复杂度指令性能（用于论文消融实验）
    complexity_stat = raw_df.groupby(["提示词方法", "复杂度分级"])["解析正确率"].mean().unstack().round(4) * 100
    complexity_stat.to_csv(Path(config.result_dir) / "llm_complexity_statistics.csv", encoding="utf-8-sig")
    logger.info("✅ 不同复杂度指令性能已保存")

    # 3.2 不同意图指令性能（用于论文泛化性分析）
    intent_stat = raw_df.groupby(["提示词方法", "意图标签"])["意图识别准确率"].mean().unstack().round(4) * 100
    intent_stat.to_csv(Path(config.result_dir) / "llm_intent_statistics.csv", encoding="utf-8-sig")
    logger.info("✅ 不同意图指令性能已保存")

    # 3.3 错误类型统计（用于论文讨论部分）
    error_stat = raw_df[raw_df["解析正确率"] == False].groupby(["提示词方法", "错误类型"]).size().reset_index(
        name="次数")
    error_stat.to_csv(Path(config.result_dir) / "llm_error_statistics.csv", encoding="utf-8-sig")
    logger.info("✅ 错误类型统计已保存")

    # ====================== 4. 打印最终实验报告 ======================
    logger.info("\n" + "=" * 80)
    logger.info("📊 LLM指令解析实验最终统计报告（顶刊级）")
    logger.info("=" * 80)

    # 4.1 核心实验指标（均值±标准差）
    logger.info("\n1. 核心实验指标（均值±标准差）:")
    logger.info(stat_df.to_string())

    # 4.2 不同提示词方法平均性能（直接用于论文表6）
    logger.info("\n2. 不同提示词方法平均性能(%):")
    prompt_perf = raw_df.groupby("提示词方法").agg({
        "解析成功率": lambda x: round(x.mean() * 100, 2),
        "意图识别准确率": lambda x: round(x.mean() * 100, 2),
        "合法JSON率": lambda x: round(x.mean() * 100, 2),
        "参数匹配F1": lambda x: round(x.mean() * 100, 2),
        "失败兜底成功率": lambda x: round(x.mean() * 100, 2)
    })
    logger.info(prompt_perf.to_string())

    # 4.3 不同复杂度指令解析正确率（用于论文消融实验）
    logger.info("\n3. 不同复杂度指令解析正确率(%):")
    logger.info(complexity_stat.round(2).to_string())

    # 4.4 不同意图指令识别准确率（用于论文泛化性分析）
    logger.info("\n4. 不同意图指令识别准确率(%):")
    logger.info(intent_stat.round(2).to_string())

    # 4.5 错误类型分布（用于论文讨论部分）
    logger.info("\n5. 主要错误类型分布(次数):")
    logger.info(error_stat.to_string())

    # 保存最终原始数据
    raw_df.to_csv(Path(config.result_dir) / "llm_experiment_raw_data_extended.csv", index=False, encoding="utf-8-sig")

    logger.info("\n✅ 所有结果生成完成！")
    logger.info("📁 原始数据: llm_experiment_results/llm_experiment_raw_data_extended.csv")
    logger.info("📊 核心统计: llm_experiment_results/llm_experiment_statistics_extended.csv")
    logger.info("📈 复杂度统计: llm_experiment_results/llm_complexity_statistics.csv")
    logger.info("🎯 意图统计: llm_experiment_results/llm_intent_statistics.csv")
    logger.info("❌ 错误统计: llm_experiment_results/llm_error_statistics.csv")
    logger.info("🖼️  可视化图表: llm_experiment_results/llm_parsing_results_extended.png")
    logger.info("📜 实验日志: llm_experiment_logs/")


# 主函数入口
if __name__ == "__main__":
    try:
        # 直接从已保存的数据生成结果（10秒完成）
        generate_results_from_temp_data()
    except Exception as e:
        logging.error(f"❌ 结果生成异常: {str(e)}", exc_info=True)
        exit(0)