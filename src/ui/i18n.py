"""Deterministic UI translation dictionary for the consumer Streamlit UI.

Scope rules (deliberate, competition-MVP safe):

* This module translates **UI-owned presentation labels only**. It never
  translates canonical regulatory text, canonical status values, or canonical
  identifiers (``rule_id`` / ``source_id`` / ``cost_id`` / ``action_id`` /
  ``attribute_id``). Those are always rendered verbatim from the approved data.
* There is no translation provider, no network access and no LLM call. The
  dictionary is a fixed literal.
* A missing key falls back to English and finally to the key itself, so a
  partial translation can never crash the UI.
"""

from __future__ import annotations

from typing import Any

DEFAULT_LANGUAGE = "en"

#: Selector labels are intentionally the same string in every language.
LANGUAGE_LABELS: dict[str, str] = {"en": "English", "zh": "中文"}

_ZH: dict[str, str] = {
    # -- header ---------------------------------------------------------- #
    "app_title": "ImportReady AI",
    "app_subtitle": "美国进口合规助手",
    "app_tagline": "在商品进入美国市场前，了解可能涉及的合规要求。",
    "disclaimer": "仅供规划参考，不构成法律建议，也不代表进口批准。",
    # -- sidebar / access ------------------------------------------------ #
    "settings_title": "模型与访问",
    "language_label": "语言",
    "appearance_label": "外观",
    "theme_light": "浅色",
    "theme_dark": "深色",
    "mode_label": "模式",
    "mode_demo": "竞赛演示模式",
    "mode_byok": "使用我自己的 API Key",
    "demo_provider_available": "竞赛演示模型已配置。",
    "demo_provider_not_configured": "本地服务器尚未配置可用于比赛演示的正式模型。",
    "demo_provider_not_verified": "已配置的模型尚未通过消费者演示认证，暂不可用于比赛演示。",
    "demo_deterministic_note": "即使没有 AI 模型，确定性合规分析仍可正常使用。",
    "provider_label": "提供商",
    "provider_none_available": "当前没有可用于消费者自带密钥的提供商。此界面仅支持已验证并开放给消费者的提供商。",
    "api_key_label": "API Key",
    "api_key_stored": "密钥仅保存在本次会话中。",
    "api_key_missing": "请输入 API Key。",
    "clear_key": "清除密钥",
    "key_cleared": "本次会话的 API Key 已清除。",
    "key_privacy_note": "仅会话内存，不写入磁盘、日志或 Git。",
    "start_over": "新商品 / 重新开始",
    "started_over": "已重置，可开始新的商品分析。",
    # -- section 1: product ---------------------------------------------- #
    "product_section": "商品",
    "product_helper": "按上架销售时的说法描述商品即可，无需专业术语。",
    "product_description": "商品描述",
    "product_placeholder": "例如：蓝牙无线耳机，带可充电锂离子电池和 USB-C 充电盒……",
    "analyze_product": "分析商品",
    "describe_required": "请先输入商品描述。",
    "sensitive_input": "请勿在商品描述中填写密钥或访问令牌。",
    # -- section 2: category confirmation -------------------------------- #
    "category_section": "品类确认",
    "ai_suggested_category": "AI 建议品类",
    "no_suggestion": "暂无 AI 品类建议。",
    "suggestion_candidate_note": "AI 建议仅为候选结果。只有您确认品类后，才会执行正式分析。",
    "confirm_category": "确认此品类",
    "change_category": "选择其他品类",
    "category_choice": "品类",
    "confirm_and_analyze": "确认品类并分析",
    "category_confirmed": "已确认品类",
    "classification_failed": "AI 品类建议暂时不可用，请手动选择品类。",
    "cat_childrens_toys": "儿童玩具",
    "cat_small_consumer_electronics": "小型消费电子产品",
    "cat_dual": "玩具与电子（双重）",
    "cat_unsupported": "暂不支持",
    "cat_uncertain": "不确定",
    "cat_unknown": "未确定",
    "cat_other": "其他",
    # -- section 3: information needed ----------------------------------- #
    "missing_section": "需要补充的信息",
    "missing_intro": "请尽量填写。未填写的项目会使对应要求的判定保持未确定状态。",
    "answer_yes": "是",
    "answer_no": "否",
    "answer_unknown": "我不知道",
    "missing_question_fallback": "请提供：{attribute}",
    "update_analysis": "更新分析",
    "missing_none": "本次分析无需补充信息。",
    "input_issues_title": "信息问题",
    "input_issues_note": "部分已提交的值无法使用，请检查并修正。",
    "value_free_text": "请填写内容",
    "value_free_list": "每行填写一项",
    # -- results --------------------------------------------------------- #
    "results_section": "分析结果",
    "summary_section": "合规摘要",
    "top_risks": "主要风险",
    "applicable_requirements": "适用要求",
    "recommended_actions": "建议行动",
    "cost_references": "成本参考",
    "what_if_section": "情景模拟",
    "evidence_sources": "证据与来源",
    "technical_details": "人工复核 / 技术细节",
    "label_category": "品类",
    "label_review_status": "复核状态",
    "label_risk_level": "评估风险等级",
    "status_analysis_completed": "分析已完成",
    "status_further_info": "需要补充信息",
    "status_human_review": "需要人工复核",
    "status_current_requirements": "已识别 {count} 项当前适用要求",
    "status_monitoring": "已识别 {count} 项监控事项",
    "status_not_assessed": "未评估：品类尚未确认。",
    "status_unsupported": "当前已批准数据不支持该品类。",
    "risk_not_assessed": "未评估",
    "risk_none": "在所评估范围内未发现风险事项",
    "risk_high": "确认的当前适用要求，需要处理",
    "risk_review": "需要补充信息或人工复核",
    "risk_monitor": "监控 / 非当前生效事项",
    "no_risk_items": "在所评估范围内未发现风险事项。",
    "risk_notes": "风险说明",
    "view_remaining": "查看其余 {count} 项",
    "key_information": "关键信息",
    "additional_information": "更多信息（{count}）",
    "review_status_NEEDS_INFO": "需要补充信息",
    "review_status_REVIEW_REQUIRED": "需要人工复核",
    "review_status_COMPLETE": "已完成",
    "review_status_UNSUPPORTED": "不支持",
    "review_status_IN_PROGRESS": "处理中",
    # -- requirements ---------------------------------------------------- #
    "requirement_type": "要求类型",
    "authority": "监管机构",
    "status": "状态",
    "source_ids": "来源编号",
    "evidence_status": "证据状态",
    "rule_status": "规则状态",
    "rule_id": "规则编号",
    "no_requirements": "在所评估范围内未确认任何当前适用要求。",
    "applicable_requirements_note": "本节仅列出已确认适用的要求。尚未解决或待复核的规则见「需要补充的信息」「建议行动」与「证据与来源」。",
    "applicability_status_APPLICABLE": "适用",
    "applicability_status_NOT_APPLICABLE": "不适用",
    "applicability_status_NEEDS_INFO": "信息不足",
    "applicability_status_REVIEW_REQUIRED": "需要人工复核",
    "applicability_status_NOT_EVALUATED": "未评估",
    "detail_show": "查看技术细节",
    # -- actions --------------------------------------------------------- #
    "group_current_obligation": "当前要求",
    "group_missing_information": "需要补充的信息",
    "group_human_review": "人工复核",
    "group_monitor": "监控事项",
    "group_cost_followup": "成本规划",
    "show_less": "收起",
    "priority": "优先级",
    "no_actions": "未生成建议行动。",
    "priority_BLOCKING": "阻塞",
    "priority_REVIEW": "复核",
    "priority_CURRENT": "当前",
    "priority_MONITOR": "监控",
    "priority_PLANNING": "规划",
    # -- cost ------------------------------------------------------------ #
    "cost_intro": "仅为品类层面的参考信息，不是到岸成本、关税/税费计算或应付费用判定。",
    "cost_status_DIRECT": "已批准的参考金额",
    "cost_status_PLANNING_ONLY": "规划估算（非报价）",
    "cost_status_QUOTE_REQUIRED": "需询价（暂无已批准金额）",
    "cost_status_DISPLAY_ONLY": "仅供参考（不参与计算）",
    "cost_amount": "参考金额",
    "cost_range": "参考区间",
    "cost_no_amount": "暂无已批准金额",
    "cost_pricing_basis": "计价方式",
    "cost_scope": "包含范围",
    "cost_exclusions": "重要排除项",
    "cost_price_status": "价格状态",
    "cost_checked": "核查日期",
    "cost_no_total_note": "不计算总额（可能包含不同币种、不同计价方式及备选记录）。",
    "cost_notes": "成本说明",
    "action_notes": "行动计划说明",
    "results_placeholder": "确认品类并完成分析后，结果将在此显示。",
    "save_key": "保存密钥",
    "no_costs": "该品类暂无已批准的成本参考记录。",
    # -- what-if --------------------------------------------------------- #
    "what_if_intro": "模拟某条结构化事实的假设变化。品类不可更改。",
    "hypothetical_scenario": "假设情景",
    "current_value": "当前",
    "what_if_value": "假设改为",
    "no_change": "不更改",
    "run_what_if": "运行情景模拟",
    "reset_what_if": "重置情景模拟",
    "what_if_not_available": "情景模拟需要已人工确认且受支持的品类。",
    "what_if_no_controls": "该品类暂无可用的假设控件。",
    "what_if_override_required": "请至少选择一项假设变化。",
    "changed_requirements": "变化的要求",
    "changed_risks": "变化的风险",
    "changed_actions": "变化的行动",
    "changed_costs": "变化的成本",
    "no_changes": "该情景未产生任何正式变化。",
    "change_type_ADDED": "新增",
    "change_type_REMOVED": "移除",
    "change_type_MODIFIED": "变更",
    "before": "变更前",
    "after": "假设后",
    "value_not_provided": "未提供",
    "what_if_notes": "情景说明",
    # -- evidence -------------------------------------------------------- #
    "evidence_intro": "正式结论所引用的已批准来源记录。",
    "no_evidence": "未引用任何已批准来源记录。",
    "source_tier": "来源层级",
    "last_checked": "最后核查",
    "canonical_url": "来源链接",
    "known_limitations": "已知数据限制",
    # -- technical ------------------------------------------------------- #
    "technical_intro": "供复核与演示使用的结构化编号与状态信息。",
    "technical_ai": "AI 运行状态",
    "technical_classification": "分类运行状态",
    "technical_knowledge": "知识快照",
    "technical_request": "请求状态",
    "technical_mode": "模式",
    "technical_status": "状态",
    "technical_stop_reason": "停止原因",
    "technical_tool_calls": "工具调用",
    "technical_error_type": "错误类型",
    "technical_schema_version": "数据版本",
    "technical_generated_at": "数据生成时间",
    "technical_collapsed_hint": "默认收起，展开可查看编号、状态与来源。",
    # -- errors ---------------------------------------------------------- #
    "err_provider_unavailable": "AI 服务暂时不可用，本次未生成合规结论。",
    "err_invalid_credential": "API Key 校验失败，请检查密钥或改用其他可用提供商。",
    "err_analysis_failed": "分析未能完成，未生成合规结论。",
    "err_what_if_input": "该假设变化无法应用，请调整后重试。",
    "err_what_if_configuration": "情景模拟内部状态不一致，未生成情景结果。",
    "err_unsupported_category": "该品类暂无已验证的合规数据。",
    "err_generic": "操作失败，请重试。",
    # =================================================================== #
    # 消费者体验 v2 —— 对话式信息录入 + 自然语言报告
    # =================================================================== #
    # -- intake ---------------------------------------------------------- #
    "intake_section": "请介绍你的商品",
    "intake_helper": "只需填写一段描述即可。像向同事介绍商品那样写明你已知的信息，不知道的内容可以留空。",
    "intake_placeholder": (
        "请描述商品，并填写你已经知道的信息，例如目标用户、材质、电池、"
        "蓝牙/Wi-Fi、充电方式、包装、已有测试或认证、销售平台、供应商资料等。"
        "不知道的内容可以不填写。"
    ),
    "intake_analyze": "分析商品",
    "intake_sensitive": "请勿在商品描述中填写密钥、API Key 或访问令牌。请删除后重试。",
    "intake_ai_unavailable": "暂时无法使用 AI 解析你的描述。你仍然可以确认类别并运行确定性分析。",
    "intake_no_text": "请先输入商品描述。",
    "intake_edit_hint": "请在上方修改商品描述，然后重新点击“分析商品”。",
    # -- AI understood --------------------------------------------------- #
    "ai_understood_title": "AI 已理解的信息",
    "ai_understood_suggested_category": "建议类别",
    "ai_understood_facts": "从你的描述中识别到的信息",
    "ai_understood_none": (
        "未能从描述中识别出具体商品信息。你可以补充描述，或直接继续，"
        "分析结果会说明哪些内容仍为未知。"
    ),
    "ai_understood_unknown_note": "你没有提到的信息仍为未知，系统不会替你假设或补全。",
    "ai_understood_you_said": "你的描述",
    "ai_understood_notes": "你提供的其他信息",
    "ai_understood_notes_note": (
        "仅作为你提供的背景信息保留，不会被视为合规事实，也不会计入任何费用参考。"
    ),
    "ai_understood_warnings": "因不明确而未采用",
    "intake_warnings_skipped": "部分表述不够明确，系统未将其作为商品事实采用。",
    "intake_warnings_count": "共有 {count} 处表述不够明确，未被采用。",
    "use_selected_category": "按此类别读取信息",
    "extraction_category_changed": (
        "类别已更改。此前按其他类别识别的信息已作废；请按当前类别重新识别后再确认。"
    ),
    "intake_bundle_stale": "该信息是按其他类别或较早的描述识别的，已作废。请确认类别后重新识别。",
    "confirm_bundle": "确认并分析",
    "edit_information": "修改商品信息",
    "confirm_category_bundle_note": "只有在你确认上述信息后，系统才会运行正式分析。",
    # -- customer report -------------------------------------------------- #
    "report_section": "你的合规报告",
    "report_overall": "总体评估",
    "report_attention": "需要注意的事项",
    "report_verify": "接下来需要确认的信息",
    "report_cost": "费用预估",
    "report_next_step": "建议的下一步",
    "report_limitations": "本评估的说明与限制",
    "report_disclaimer": (
        "本报告仅根据你提供的信息和已核准的参考数据自动生成，用于初步筛查，"
        "不构成法律意见，也不代表批准或通关许可。"
    ),
    "report_attention_none": "在当前评估范围内，尚未确认存在当前适用的合规要求。",
    "report_verify_lead": "接下来最值得确认的信息：",
    "report_verify_none": "当前评估范围内无需补充更多商品信息。",
    "report_verify_more": "另有 {count} 项列在“专业详情”中。",
    "report_limitations_scope": "本评估仅覆盖 ImportReady 已核准数据中的类别与规则，并基于你提供的信息。",
    "report_limitations_unstated": "你未提供的细节仍为未知，因此部分要求可能仍处于未确定状态。",
    "report_limitations_review": "在用于商业决策前，仍需人工复核。",
    # overall assessment variants (bounded risk language only - no scores)
    "report_overall_attention": (
        "根据当前已有的信息，已识别出一项或多项当前适用的合规要求。"
        "在将该商品视为可进口或可销售之前，应先处理这些事项。"
    ),
    "report_overall_verify": (
        "根据当前已有的信息，仍有若干合规要点需要核实。"
        "目前尚无法对该商品作出高置信度的评估，建议补充商品或文件信息。"
    ),
    "report_overall_lower": (
        "在当前评估范围内，尚未确认存在重大的当前合规问题。"
        "部分事项仍可能需要核实，本评估不应被视为法律批准或通关许可。"
    ),
    "report_overall_monitor": (
        "仍有一些非当前或需持续关注的事项值得跟踪，但它们不被视为当前已确认的义务。"
    ),
    "report_overall_unsupported": (
        "该商品似乎不属于本 MVP 当前覆盖的类别。当前版本支持儿童玩具和小型消费电子产品。"
    ),
    "report_overall_not_assessed": "商品类别尚未确认，因此尚未运行合规评估。",
    # compliance areas (closed, canonical-derived vocabulary)
    "area_wireless": "无线／射频（RF）相关要求",
    "area_battery": "电池相关要求",
    "area_children": "儿童产品相关要求",
    "area_labeling": "标签与文件要求",
    "area_testing": "测试与认证",
    "area_monitoring": "监测中／提案中的要求",
    "area_general": "其他适用要求",
    "report_area_lead": "当前涉及的主要方面：",
    # -- cost outlook ----------------------------------------------------- #
    "report_cost_none": "当前评估范围内暂无已核准的合规费用参考。",
    "report_cost_generic": (
        "现有的合规费用参考显示，可能涉及测试或文件方面的支出。"
        "其中部分项目有规划性估算，另一些则需要供应商或实验室报价。"
    ),
    "report_cost_amount": "其中一条可用的规划参考显示，此类活动的费用约为 {amount}。",
    "report_cost_quote_required": "{count} 个项目没有已核准金额，需要供应商或实验室报价。",
    "report_cost_no_total": (
        "目前无法给出可靠的到岸总成本估算，因为采购价格、运费、关税／税率处理"
        "以及其他与交易相关的费用尚未纳入当前评估模型。"
    ),
    "report_cost_unassessed": "该商品尚未进行费用参考评估。",
    "report_commercial_reported": "你提供的信息：{note}",
    "report_commercial_not_additive": "你提供的金额与上述合规费用参考不会自动相加。",
    # -- recommended next step -------------------------------------------- #
    "report_next_attention": "请先处理上述当前适用的合规要求，并在商业使用前保留相应证明文件。",
    "report_next_verify": "在做出最终进口决定前，请先确认“接下来需要确认的信息”中所列内容。",
    "report_next_lower": (
        "当前信息显示，在已评估范围内的风险相对较低，但在用于商业决策前仍建议进行当地或法律复核。"
    ),
    "report_next_monitor": "请持续跟踪监测中的事项，并在商品上市前再次核查。",
    "report_next_unsupported": "该商品不在支持范围内，因此不生成合规下一步建议。",
    # -- optional advanced sections --------------------------------------- #
    "technical_rule_results": "规则判定明细",
    "technical_rule_results_none": "当前范围内没有规则判定结果。",
    "technical_details_label": "专业详情",
    "scenario_section": "情景模拟",
    "scenario_lead": "可选。修改一个已确认的细节，查看评估结果会如何变化。",
    "scenario_summary_lead": "如果你设定的情景成立：",
    "scenario_summary_item": "{attribute} → {value}",
    "scenario_summary_empty": "该情景不会改变已评估的要求。",
}


_EN: dict[str, str] = {
    # -- header ---------------------------------------------------------- #
    "app_title": "ImportReady AI",
    "app_subtitle": "US Import Compliance Assistant",
    "app_tagline": "Understand what your product may need before entering the U.S. market.",
    "disclaimer": "For planning support only. This is not legal advice or import approval.",
    # -- sidebar / access ------------------------------------------------ #
    "settings_title": "Model & access",
    "language_label": "Language",
    "appearance_label": "Appearance",
    "theme_light": "Light",
    "theme_dark": "Dark",
    "mode_label": "Mode",
    "mode_demo": "Competition Demo",
    "mode_byok": "Use My Own API Key",
    "demo_provider_available": "Competition Demo model is configured.",
    "demo_provider_not_configured": (
        "Competition Demo provider is not configured on this local server."
    ),
    "demo_provider_not_verified": (
        "The configured model is not certified for consumer demo use."
    ),
    "demo_deterministic_note": (
        "Deterministic compliance analysis still works without an AI model."
    ),
    "provider_label": "Provider",
    "provider_none_available": (
        "No provider is currently available for consumer key use. Only verified, "
        "consumer-enabled providers can be used here."
    ),
    "api_key_label": "API Key",
    "api_key_stored": "A key is stored for this session only.",
    "api_key_missing": "Enter an API key.",
    "clear_key": "Clear Key",
    "key_cleared": "API key cleared for this session.",
    "key_privacy_note": "Session only. Never written to disk, logs, or Git.",
    "start_over": "New Product / Start Over",
    "started_over": "Cleared. You can start a new product analysis.",
    # -- section 1: product ---------------------------------------------- #
    "product_section": "Product",
    "product_helper": (
        "Describe the product the way you would list it for sale. Plain language is fine."
    ),
    "product_description": "Product Description",
    "product_placeholder": (
        "e.g. Bluetooth wireless earbuds with a rechargeable lithium-ion battery "
        "and a USB-C charging case..."
    ),
    "analyze_product": "Analyze Product",
    "describe_required": "Enter a product description to begin.",
    "sensitive_input": (
        "Do not enter credentials or access tokens in the product description."
    ),
    # -- section 2: category confirmation -------------------------------- #
    "category_section": "Category Confirmation",
    "ai_suggested_category": "AI Suggested Category",
    "no_suggestion": "No AI category suggestion is available.",
    "suggestion_candidate_note": (
        "The AI suggestion is a candidate only. Canonical analysis runs only after "
        "you confirm a category."
    ),
    "confirm_category": "Confirm this category",
    "change_category": "Choose a different category",
    "category_choice": "Category",
    "confirm_and_analyze": "Confirm Category and Analyze",
    "category_confirmed": "Category confirmed",
    "classification_failed": (
        "AI category suggestion is temporarily unavailable. Please select a category "
        "manually."
    ),
    "cat_childrens_toys": "Children's Toys",
    "cat_small_consumer_electronics": "Small Consumer Electronics",
    "cat_dual": "Toys and Electronics (Dual)",
    "cat_unsupported": "Not supported",
    "cat_uncertain": "Uncertain",
    "cat_unknown": "Not determined",
    "cat_other": "Other",
    # -- section 3: information needed ----------------------------------- #
    "missing_section": "Information Needed",
    "missing_intro": (
        "Answer what you can. Unanswered items leave those requirements undetermined."
    ),
    "answer_yes": "Yes",
    "answer_no": "No",
    "answer_unknown": "I don't know",
    "missing_question_fallback": "Please provide: {attribute}",
    "update_analysis": "Update Analysis",
    "missing_none": "No missing information is required for this analysis.",
    "input_issues_title": "Information problems",
    "input_issues_note": "Some supplied values could not be used. Please review and correct them.",
    "value_free_text": "Enter a value",
    "value_free_list": "One item per line",
    # -- results --------------------------------------------------------- #
    "results_section": "Results",
    "summary_section": "Compliance Summary",
    "top_risks": "Top Risks",
    "applicable_requirements": "Applicable Requirements",
    "recommended_actions": "Recommended Actions",
    "cost_references": "Cost References",
    "what_if_section": "What-if Analysis",
    "evidence_sources": "Evidence & Sources",
    "technical_details": "Human Review / Technical Details",
    "label_category": "Category",
    "label_review_status": "Review status",
    "label_risk_level": "Assessed risk level",
    "status_analysis_completed": "Analysis completed",
    "status_further_info": "Further information required",
    "status_human_review": "Human review required",
    "status_current_requirements": "{count} current applicable requirement(s) identified",
    "status_monitoring": "{count} monitoring item(s) identified",
    "status_not_assessed": "Not assessed: the category is not confirmed.",
    "status_unsupported": "The current approved data does not support this category.",
    "risk_not_assessed": "Not assessed",
    "risk_none": "No risk-bearing item found in the assessed scope",
    "risk_high": "Confirmed current applicable requirement requiring attention",
    "risk_review": "Needs further information or human review",
    "risk_monitor": "Monitor / non-current item",
    "no_risk_items": "No risk-bearing item was found in the assessed scope.",
    "risk_notes": "Assessment notes",
    "view_remaining": "View remaining {count}",
    "key_information": "Key information",
    "additional_information": "Additional information ({count})",
    "review_status_NEEDS_INFO": "Further information required",
    "review_status_REVIEW_REQUIRED": "Human review required",
    "review_status_COMPLETE": "Completed",
    "review_status_UNSUPPORTED": "Not supported",
    "review_status_IN_PROGRESS": "In progress",
    # -- requirements ---------------------------------------------------- #
    "requirement_type": "Requirement type",
    "authority": "Authority",
    "status": "Status",
    "source_ids": "Sources",
    "evidence_status": "Evidence status",
    "rule_status": "Rule status",
    "rule_id": "Rule ID",
    "no_requirements": "No current applicable requirement has been confirmed in the assessed scope.",
    "applicable_requirements_note": (
        "This section lists only requirements confirmed as applicable. Unresolved or "
        "review-pending rules appear under Information Needed, Recommended Actions and "
        "Evidence & Sources."
    ),
    "applicability_status_APPLICABLE": "Applicable",
    "applicability_status_NOT_APPLICABLE": "Not applicable",
    "applicability_status_NEEDS_INFO": "Information incomplete",
    "applicability_status_REVIEW_REQUIRED": "Human review required",
    "applicability_status_NOT_EVALUATED": "Not evaluated",
    "detail_show": "Technical details",
    # -- actions --------------------------------------------------------- #
    "group_current_obligation": "Current Requirements",
    "group_missing_information": "Information Needed",
    "group_human_review": "Human Review",
    "group_monitor": "Monitoring",
    "group_cost_followup": "Cost Planning",
    "show_less": "Show less",
    "priority": "Priority",
    "no_actions": "No recommended action was produced.",
    "priority_BLOCKING": "Blocking",
    "priority_REVIEW": "Review",
    "priority_CURRENT": "Current",
    "priority_MONITOR": "Monitor",
    "priority_PLANNING": "Planning",
    # -- cost ------------------------------------------------------------ #
    "cost_intro": (
        "Category-level reference information only. This is not a landed cost, a duty or "
        "tax calculation, or a payable-cost determination."
    ),
    "cost_status_DIRECT": "Approved reference amount",
    "cost_status_PLANNING_ONLY": "Planning estimate (not a quote)",
    "cost_status_QUOTE_REQUIRED": "Quote required (no approved amount)",
    "cost_status_DISPLAY_ONLY": "Informational (non-calculating)",
    "cost_amount": "Reference amount",
    "cost_range": "Reference range",
    "cost_no_amount": "No approved numeric amount",
    "cost_pricing_basis": "Pricing basis",
    "cost_scope": "Scope included",
    "cost_exclusions": "Important exclusions",
    "cost_price_status": "Price status",
    "cost_checked": "Checked",
    "cost_no_total_note": (
        "No total is calculated (records may use different currencies, pricing bases and "
        "alternative or add-on items)."
    ),
    "no_costs": "No approved cost reference record exists for this category.",
    "cost_notes": "Cost notes",
    "action_notes": "Action plan notes",
    "results_placeholder": (
        "Confirm the category and complete the analysis to see results here."
    ),
    "save_key": "Save Key",
    # -- what-if --------------------------------------------------------- #
    "what_if_intro": (
        "Test a hypothetical structured fact change. The product category cannot be changed."
    ),
    "hypothetical_scenario": "Hypothetical Scenario",
    "current_value": "Current",
    "what_if_value": "What if",
    "no_change": "No change",
    "run_what_if": "Run What-if",
    "reset_what_if": "Reset What-if",
    "what_if_not_available": "What-if requires a human-confirmed, supported category.",
    "what_if_no_controls": (
        "No supported hypothetical control is available for this category."
    ),
    "what_if_override_required": "Select at least one hypothetical change.",
    "changed_requirements": "Changed Requirements",
    "changed_risks": "Changed Risks",
    "changed_actions": "Changed Actions",
    "changed_costs": "Changed Costs",
    "no_changes": "No canonical change was produced by this scenario.",
    "change_type_ADDED": "Added",
    "change_type_REMOVED": "Removed",
    "change_type_MODIFIED": "Modified",
    "before": "Before",
    "after": "Hypothetical after",
    "value_not_provided": "Not provided",
    "what_if_notes": "Scenario notes",
    # -- evidence -------------------------------------------------------- #
    "evidence_intro": "Approved source records referenced by the canonical findings.",
    "no_evidence": "No approved source record is referenced.",
    "source_tier": "Source tier",
    "last_checked": "Last checked",
    "canonical_url": "Source URL",
    "known_limitations": "Known data limitations",
    # -- technical ------------------------------------------------------- #
    "technical_intro": (
        "Structured identifiers and technical status, for review and demo purposes."
    ),
    "technical_ai": "AI runtime",
    "technical_classification": "Classification runtime",
    "technical_knowledge": "Knowledge snapshot",
    "technical_request": "Request status",
    "technical_mode": "Mode",
    "technical_status": "Status",
    "technical_stop_reason": "Stop reason",
    "technical_tool_calls": "Tool calls",
    "technical_error_type": "Error type",
    "technical_schema_version": "Schema version",
    "technical_generated_at": "Data generated at",
    "technical_collapsed_hint": (
        "Collapsed by default. Expand to inspect identifiers, statuses and sources."
    ),
    # -- errors ---------------------------------------------------------- #
    "err_provider_unavailable": (
        "AI service is temporarily unavailable. No compliance conclusion was generated."
    ),
    "err_invalid_credential": (
        "The API key could not be verified. Please check the key or use another "
        "available provider."
    ),
    "err_analysis_failed": (
        "Analysis could not be completed. No compliance conclusion was generated."
    ),
    "err_what_if_input": "This hypothetical change could not be applied.",
    "err_what_if_configuration": (
        "The scenario could not be produced because its canonical inputs are inconsistent."
    ),
    "err_unsupported_category": (
        "Verified compliance data is not available for this category."
    ),
    "err_generic": "Something went wrong. Please try again.",
    # =================================================================== #
    # Consumer UX v2 — conversational intake + narrative report
    # =================================================================== #
    # -- intake ---------------------------------------------------------- #
    "intake_section": "Tell us about your product",
    "intake_helper": (
        "One description is enough. Write it the way you would explain it to a "
        "colleague, and leave out anything you do not know."
    ),
    "intake_placeholder": (
        "Describe the product and include anything you already know - intended users, "
        "materials, battery, Bluetooth/Wi-Fi, charger, packaging, testing, "
        "certificates, marketplace, supplier information, or other relevant details. "
        "You can leave out anything you don't know."
    ),
    "intake_analyze": "Analyze Product",
    "intake_sensitive": (
        "Do not enter credentials, API keys or access tokens in the product "
        "description. Remove them and try again."
    ),
    "intake_ai_unavailable": (
        "AI reading of your description is temporarily unavailable. You can still "
        "confirm a category and run the deterministic analysis."
    ),
    "intake_no_text": "Enter a product description to begin.",
    "intake_edit_hint": (
        "Update your description above, then select Analyze Product again."
    ),
    # -- AI understood --------------------------------------------------- #
    "ai_understood_title": "AI understood",
    "ai_understood_suggested_category": "Suggested category",
    "ai_understood_facts": "Information found in your description",
    "ai_understood_none": (
        "No product details were read from your description. Add details, or continue "
        "and the analysis will state what is still unknown."
    ),
    "ai_understood_unknown_note": (
        "Anything you did not state stays unknown. It is never assumed or filled in "
        "for you."
    ),
    "ai_understood_you_said": "you said",
    "ai_understood_notes": "Other information you provided",
    "ai_understood_notes_note": (
        "Kept as your own context only. It is not treated as a compliance fact and is "
        "not added to any cost reference."
    ),
    "ai_understood_warnings": "Skipped as unclear",
    "intake_warnings_skipped": (
        "Some ambiguous information was skipped because it could not be confirmed "
        "from your description."
    ),
    "intake_warnings_count": "{count} ambiguous item(s) were skipped.",
    "use_selected_category": "Read details for this category",
    "extraction_category_changed": (
        "The category changed. Details read for the previous category were discarded; "
        "read them again for this category before confirming."
    ),
    "intake_bundle_stale": (
        "That information was read for a different category or an older description, "
        "so it was discarded. Confirm the category to read it again."
    ),
    "confirm_bundle": "Confirm and analyze",
    "edit_information": "Edit product information",
    "confirm_category_bundle_note": (
        "Canonical analysis runs only after you confirm this bundle."
    ),
    # -- customer report -------------------------------------------------- #
    "report_section": "Your compliance report",
    "report_overall": "Overall assessment",
    "report_attention": "What needs attention",
    "report_verify": "What to verify next",
    "report_cost": "Cost outlook",
    "report_next_step": "Recommended next step",
    "report_limitations": "Limitations of this assessment",
    "report_disclaimer": (
        "This report is an automated screening aid based only on the information you "
        "provided and the approved reference data. It is not legal advice, not an "
        "approval, and not import clearance."
    ),
    "report_attention_none": (
        "No current applicable compliance requirement has been confirmed in the "
        "assessed scope."
    ),
    "report_verify_lead": "The most useful details to confirm next:",
    "report_verify_none": (
        "No further product detail is required for the assessed scope."
    ),
    "report_verify_more": (
        "{count} further item(s) are listed in Technical details."
    ),
    "report_limitations_scope": (
        "The assessment covers only the categories and rules in the approved "
        "ImportReady data set, using the details you provided."
    ),
    "report_limitations_unstated": (
        "Details you did not provide remain unknown, so some requirements may stay "
        "undetermined."
    ),
    "report_limitations_review": (
        "Human review remains required before commercial reliance."
    ),
    # overall assessment variants (bounded risk language only - no scores)
    "report_overall_attention": (
        "Based on the information currently available, one or more current applicable "
        "compliance requirements have been identified. These items should be addressed "
        "before relying on the product as ready for U.S. import or sale."
    ),
    "report_overall_verify": (
        "Based on the information currently available, several compliance points still "
        "require verification. The product cannot yet be assessed with high confidence, "
        "and additional product or documentation details are recommended."
    ),
    "report_overall_lower": (
        "Within the currently assessed scope, no major current compliance issue has "
        "been confirmed. Some items may still require verification, and this assessment "
        "should not be treated as legal approval or import clearance."
    ),
    "report_overall_monitor": (
        "Some non-current or monitoring items remain relevant and should be tracked, "
        "but they are not treated as current confirmed obligations."
    ),
    "report_overall_unsupported": (
        "This product does not appear to fall within the categories currently covered "
        "by this MVP. The current version supports children's toys and small consumer "
        "electronics."
    ),
    "report_overall_not_assessed": (
        "The product category is not confirmed yet, so no compliance assessment has "
        "been run."
    ),
    # compliance areas (closed, canonical-derived vocabulary)
    "area_wireless": "wireless / RF requirements",
    "area_battery": "battery-related requirements",
    "area_children": "children's product requirements",
    "area_labeling": "labeling and documentation",
    "area_testing": "testing and certification",
    "area_monitoring": "monitoring / proposed requirements",
    "area_general": "other applicable requirements",
    "report_area_lead": "The areas currently flagged are:",
    # -- cost outlook ----------------------------------------------------- #
    "report_cost_none": (
        "No approved compliance cost reference is available for the assessed scope yet."
    ),
    "report_cost_generic": (
        "Available compliance cost references suggest that testing or documentation "
        "expenses may apply. Some items have planning estimates, while others require a "
        "supplier or laboratory quote."
    ),
    "report_cost_amount": (
        "One available planning reference indicates approximately {amount} for this "
        "type of activity."
    ),
    "report_cost_quote_required": (
        "{count} item(s) have no approved amount and require a supplier or laboratory "
        "quote."
    ),
    "report_cost_no_total": (
        "A reliable total landed-cost estimate cannot be calculated because purchase "
        "price, shipping, tariff/duty treatment and other transaction-specific costs "
        "are not modelled in the current assessment."
    ),
    "report_cost_unassessed": (
        "Cost references were not assessed for this product yet."
    ),
    "report_commercial_reported": "You reported: {note}",
    "report_commercial_not_additive": (
        "Amounts you provided and the compliance cost references above are not "
        "automatically additive."
    ),
    # -- recommended next step -------------------------------------------- #
    "report_next_attention": (
        "Address the current applicable requirement(s) above and keep the supporting "
        "documentation on file before commercial reliance."
    ),
    "report_next_verify": (
        "Confirm the items listed under \"What to verify next\" before making a final "
        "import decision."
    ),
    "report_next_lower": (
        "Current information suggests a relatively lower level of identified risk "
        "within the assessed scope, but local/legal review is recommended before "
        "commercial reliance."
    ),
    "report_next_monitor": (
        "Track the monitoring items and re-check them before the product is marketed."
    ),
    "report_next_unsupported": (
        "This product is outside the supported scope, so no compliance next step is "
        "produced."
    ),
    # -- optional advanced sections --------------------------------------- #
    "technical_rule_results": "Canonical rule results",
    "technical_rule_results_none": "No rule result is available for this scope.",
    "technical_details_label": "Technical details",
    "scenario_section": "Explore a scenario",
    "scenario_lead": (
        "Optional. Change one confirmed detail to see how the assessed requirements "
        "would move."
    ),
    "scenario_summary_lead": "If your scenario applies:",
    "scenario_summary_item": "{attribute} -> {value}",
    "scenario_summary_empty": "This scenario does not change the assessed requirements.",
}

#: Fixed translation table. ``en`` is the fallback language for every key.
TRANSLATIONS: dict[str, dict[str, str]] = {"en": _EN, "zh": _ZH}


# --------------------------------------------------------------------------- #
# Per-attribute clarification questions (customer-facing interaction copy)
# --------------------------------------------------------------------------- #
#: One approved bilingual question per canonical ``attribute_id``.
#:
#: These are the **customer interaction prompts** for the missing-information
#: flow. They are keyed by ``attribute_id`` - never by ``rule_id`` - because a
#: canonical rule's ``clarification_question`` is rule-scoped prose that covers
#: several attributes at once and must never be shown as one attribute's prompt.
#:
#: Only this customer-facing question copy is localized. Canonical identifiers
#: (``rule_id``, ``attribute_id``, ``source_id``, ``cost_id``), canonical
#: enum/status values, regulatory titles, evidence text and legal requirement
#: prose always stay verbatim.
ATTRIBUTE_QUESTIONS: dict[str, dict[str, str]] = {
    # -- common ---------------------------------------------------------- #
    "A-CMN-001": {
        "en": "Which product category best describes this product?",
        "zh": "该商品属于哪个商品类别？",
    },
    "A-CMN-002": {
        "en": "In which country was the product manufactured or substantially transformed?",
        "zh": "该商品的制造国或实质性改变发生地是哪个国家/地区？",
    },
    "A-CMN-003": {
        "en": "In which U.S. states or territories will the product be sold?",
        "zh": "该商品将在美国哪些州或地区销售？",
    },
    "A-CMN-004": {
        "en": "On which marketplaces will the product be listed?",
        "zh": "该商品将在哪些平台上架销售？",
    },
    "A-CMN-005": {
        "en": (
            "Have the material, design, process, component source, or factory changed "
            "since the last compliance check?"
        ),
        "zh": "自上次合规确认以来，材料、设计、工艺、零部件来源或工厂是否发生变更？",
    },
    "A-CMN-006": {
        "en": "What processes and locations support the substantial-transformation claim?",
        "zh": "支持实质性改变认定的加工工序和完成地点是什么？",
    },
    "A-CMN-007": {
        "en": "Where will the country-of-origin marking appear on the product or packaging?",
        "zh": "原产地标识将标注在商品或包装的什么位置？",
    },
    "A-CMN-008": {
        "en": "Are you claiming a country-of-origin marking exception?",
        "zh": "是否主张原产地标识的豁免情形？",
    },
    "A-CMN-009": {
        "en": "How many employees does the business have?",
        "zh": "该企业的员工人数是多少？",
    },
    "A-CMN-010": {
        "en": "Is there a known exposure to a California Proposition 65 listed chemical?",
        "zh": "是否存在加州 65 号提案所列化学物质的已知暴露？",
    },
    "A-CMN-011": {
        "en": "What safe-harbor or exposure assessment supports the Proposition 65 conclusion?",
        "zh": "支持 65 号提案结论的安全港或暴露评估依据是什么？",
    },
    "A-CMN-012": {
        "en": "In which Amazon store country will the product be listed?",
        "zh": "该商品将在哪个亚马逊站点国家上架？",
    },
    "A-CMN-013": {
        "en": "What is the ASIN of the listing?",
        "zh": "该商品链接的 ASIN 是什么？",
    },
    "A-CMN-014": {
        "en": "Has Amazon requested compliance documents for this listing?",
        "zh": "亚马逊是否已就该商品要求提供合规文件？",
    },
    "A-CMN-015": {
        "en": "What is the date of the existing test report?",
        "zh": "现有检测报告的日期是哪一天？",
    },
    "A-CMN-016": {
        "en": "Does the product match a recall or a platform prohibition list?",
        "zh": "该商品是否涉及召回或平台禁售清单？",
    },
    # -- small consumer electronics -------------------------------------- #
    "A-ELEC-001": {
        "en": "Is this electronic device designed or marketed for children?",
        "zh": "该电子产品是否面向儿童设计或销售？",
    },
    "A-ELEC-002": {
        "en": "Does the device intentionally transmit RF?",
        "zh": "该设备是否会主动发射射频（RF）信号？",
    },
    "A-ELEC-003": {
        "en": "Which radio protocols and frequency bands does the device use?",
        "zh": "该设备使用哪些无线协议和频段？",
    },
    "A-ELEC-004": {
        "en": "Is the product a digital device, peripheral, or switching power supply?",
        "zh": "该商品是否属于数字设备、外设或开关电源？",
    },
    "A-ELEC-005": {
        "en": "What is the highest frequency the device generates or uses, in MHz?",
        "zh": "设备产生或使用的最高频率是多少 MHz？",
    },
    "A-ELEC-006": {
        "en": "How is the device powered during normal operation?",
        "zh": "设备在正常工作时采用哪些供电方式？",
    },
    "A-ELEC-007": {
        "en": "Is the radio a certified module, a limited modular approval, or a custom radio?",
        "zh": "该射频方案是已认证模块、有限模块化认证，还是自研射频？",
    },
    "A-ELEC-008": {
        "en": (
            "What is the module FCC ID, and which antenna, power, host, and co-location "
            "conditions does its grant allow?"
        ),
        "zh": "模块的 FCC ID 是什么？其认证许可允许哪些天线、功率、主机及同址安装条件？",
    },
    "A-ELEC-009": {
        "en": "What is the minimum separation between the transmitter and the body, in cm?",
        "zh": "发射器与人体的最小间距是多少厘米？",
    },
    "A-ELEC-010": {
        "en": "How many transmitters can operate simultaneously?",
        "zh": "可同时工作的发射器数量是多少？",
    },
    "A-ELEC-011": {
        "en": "What battery chemistry does the product use?",
        "zh": "该商品使用何种电池化学体系？",
    },
    "A-ELEC-012": {
        "en": "How is the lithium battery shipped (contained in, packed with, alone, or spare)?",
        "zh": "锂电池以何种方式运输（装在设备内、与设备同包、单独或备用）？",
    },
    "A-ELEC-013": {
        "en": "What is the lithium-ion battery rating in watt-hours (Wh)?",
        "zh": "锂离子电池的额定能量是多少瓦时（Wh）？",
    },
    "A-ELEC-014": {
        "en": "What is the lithium-metal content, in grams?",
        "zh": "锂金属含量是多少克？",
    },
    "A-ELEC-015": {
        "en": "Is a UN 38.3 test summary available?",
        "zh": "是否具备 UN 38.3 检测概要？",
    },
    "A-ELEC-016": {
        "en": "Does the product contain or use a button/coin cell?",
        "zh": "该商品是否包含或使用纽扣电池？",
    },
    "A-ELEC-017": {
        "en": "What are the button/coin cell diameter, height, chemistry, and model?",
        "zh": "纽扣电池的直径、高度、化学体系和型号分别是什么？",
    },
    "A-ELEC-018": {
        "en": "Is an external power supply included or sold with the product?",
        "zh": "该商品是否随附或搭配销售外接电源？",
    },
    "A-ELEC-019": {
        "en": "Does the product charge a battery?",
        "zh": "该商品是否具备电池充电功能？",
    },
    "A-ELEC-020": {
        "en": "What are the mains input voltage, frequency, current, plug type, and protection class?",
        "zh": "市电输入的电压、频率、电流、插头类型和保护等级分别是什么？",
    },
    "A-ELEC-021": {
        "en": "What is the result of the FCC Covered List screen?",
        "zh": "FCC 涵盖清单（Covered List）筛查结果是什么？",
    },
    "A-ELEC-022": {
        "en": "Is the device marketed for residential or business use?",
        "zh": "该设备面向住宅环境还是商业环境使用？",
    },
    "A-ELEC-023": {
        "en": "Which FCC authorization path applies (certification, SDoC, or none)?",
        "zh": "适用哪种 FCC 授权路径（认证、SDoC 或无需授权）？",
    },
    "A-ELEC-024": {
        "en": "Who is the U.S.-located responsible party (name and address)?",
        "zh": "美国境内的责任方是谁（名称和地址）？",
    },
    "A-ELEC-025": {
        "en": "What is the FCC ID?",
        "zh": "FCC ID 是什么？",
    },
    "A-ELEC-026": {
        "en": "Is the device Class A or Class B?",
        "zh": "该设备属于 A 类还是 B 类？",
    },
    "A-ELEC-027": {
        "en": "Will compliance information appear on a physical label or an electronic label?",
        "zh": "合规信息采用实体标签还是电子标签？",
    },
    "A-ELEC-028": {
        "en": "What conducted and radiated power levels were measured?",
        "zh": "测得的传导和辐射功率分别是多少？",
    },
    "A-ELEC-029": {
        "en": "What antenna type and gain does each transmitter use?",
        "zh": "各发射器使用的天线类型和增益是多少？",
    },
    "A-ELEC-030": {
        "en": "What is the transmitter duty cycle?",
        "zh": "发射器的工作占空比是多少？",
    },
    "A-ELEC-031": {
        "en": "What is the external power supply output type and rated power?",
        "zh": "外接电源的输出类型和额定功率是什么？",
    },
    "A-ELEC-032": {
        "en": "Are you claiming an external power supply exception?",
        "zh": "是否主张外接电源的豁免情形？",
    },
    "A-ELEC-033": {
        "en": "What is the source-rated battery energy?",
        "zh": "电源端标称的电池能量是多少？",
    },
    "A-ELEC-034": {
        "en": "Is a medical-device exclusion claimed for the charger?",
        "zh": "该充电器是否主张医疗器械豁免？",
    },
    "A-ELEC-035": {
        "en": "How many batteries or cells are in the package?",
        "zh": "包装内电池或电芯的数量是多少？",
    },
    "A-ELEC-036": {
        "en": "What is the documented basis for concluding that no FCC authorization is required?",
        "zh": "判定无需 FCC 授权的书面依据是什么？",
    },
    "A-ELEC-037": {
        "en": "Is an e-mobility exclusion claimed for this product?",
        "zh": "该商品是否主张电动出行设备豁免？",
    },
    "A-ELEC-038": {
        "en": "Is a loose 18650 cell included or sold?",
        "zh": "是否包含或销售裸装 18650 电芯？",
    },
    "A-ELEC-039": {
        "en": "Is the cell a zinc-air cell?",
        "zh": "该电池是否为锌空气电池？",
    },
    "A-ELEC-040": {
        "en": "Is the button/coin cell installed, separately packaged, or both?",
        "zh": "纽扣电池是已安装、单独包装，还是两者都有？",
    },
    "A-ELEC-041": {
        "en": "Does the product follow the toy path or the non-toy path?",
        "zh": "该商品适用玩具路径还是非玩具路径？",
    },
    # -- children's toys -------------------------------------------------- #
    "A-TOY-001": {
        "en": "What is the minimum intended age, in months?",
        "zh": "目标适用年龄的下限是多少个月？",
    },
    "A-TOY-002": {
        "en": "What is the maximum intended age, in years?",
        "zh": "目标适用年龄的上限是多少岁？",
    },
    "A-TOY-003": {
        "en": (
            "Which child-appeal factors apply (labeling, packaging, advertising, size, "
            "theme, play value)?"
        ),
        "zh": "存在哪些吸引儿童的因素（标识、包装、广告、尺寸、主题、玩耍价值）？",
    },
    "A-TOY-004": {
        "en": "Does any part fit the small-parts cylinder as received?",
        "zh": "在收到状态下，是否有部件可完全放入小零件测试筒？",
    },
    "A-TOY-005": {
        "en": "Does any part fit the small-parts cylinder after use-and-abuse testing?",
        "zh": "经使用和滥用测试后，是否有部件可完全放入小零件测试筒？",
    },
    "A-TOY-006": {
        "en": "Does the toy contain a small ball, marble, or latex balloon?",
        "zh": "该玩具是否包含小球、弹珠或乳胶气球？",
    },
    "A-TOY-007": {
        "en": "List every accessible component with its material and supplier.",
        "zh": "请列出所有可触及部件及其材料和供应商。",
    },
    "A-TOY-008": {
        "en": "Does any accessible component have paint or a similar surface coating?",
        "zh": "是否有可触及部件带有油漆或类似表面涂层？",
    },
    "A-TOY-009": {
        "en": "What are the coating type and the homogeneous colors?",
        "zh": "涂层的类型和各均质颜色分别是什么？",
    },
    "A-TOY-010": {
        "en": "Does any accessible component contain plasticizers (phthalates)?",
        "zh": "是否有可触及部件含有增塑剂（邻苯二甲酸酯）？",
    },
    "A-TOY-011": {
        "en": "Does the material require testing, qualify for a determination/exemption, or is it inaccessible?",
        "zh": "该材料需要检测、可申请认定/豁免，还是不可触及？",
    },
    "A-TOY-012": {
        "en": "Does the toy contain magnets?",
        "zh": "该玩具是否含有磁体？",
    },
    "A-TOY-013": {
        "en": "Can a small magnet become loose or separable after use-and-abuse testing?",
        "zh": "经使用和滥用测试后，小磁体是否会松脱或分离？",
    },
    "A-TOY-014": {
        "en": "What is the magnet flux index (kG² mm²)?",
        "zh": "磁体的磁通量指数是多少（kG² mm²）？",
    },
    "A-TOY-015": {
        "en": "Is the toy battery operated?",
        "zh": "该玩具是否使用电池驱动？",
    },
    "A-TOY-016": {
        "en": "Does the toy contain or use a button/coin cell?",
        "zh": "该玩具是否包含或使用纽扣电池？",
    },
    "A-TOY-017": {
        "en": "How is the battery compartment opened or the battery replaced?",
        "zh": "电池仓如何开启或电池如何更换？",
    },
    "A-TOY-018": {
        "en": "Is the button/coin cell supplied separately with the toy?",
        "zh": "纽扣电池是否与玩具分开包装提供？",
    },
    "A-TOY-019": {
        "en": "Is the article intended to connect to a nominal 120 V branch circuit?",
        "zh": "该产品是否设计为接入标称 120 V 分支电路？",
    },
    "A-TOY-020": {
        "en": "Which toy feature types are present?",
        "zh": "该玩具包含哪些功能类型？",
    },
    "A-TOY-021": {
        "en": "Does the toy contain water beads or another expanding material?",
        "zh": "该玩具是否含有水珠或其他膨胀材料？",
    },
    "A-TOY-022": {
        "en": "Is the product a neck float?",
        "zh": "该商品是否为颈部浮圈？",
    },
    "A-TOY-023": {
        "en": "Is the product in continuous production?",
        "zh": "该产品是否处于持续生产状态？",
    },
    "A-TOY-024": {
        "en": "Which children's product safety rules apply to the finished product?",
        "zh": "成品适用哪些儿童产品安全规则？",
    },
    "A-TOY-025": {
        "en": "Is an exclusion from 16 CFR part 1501 claimed?",
        "zh": "是否主张 16 CFR 1501 的排除适用？",
    },
    "A-TOY-026": {
        "en": "What is the circuit voltage, in volts RMS?",
        "zh": "电路电压是多少伏（RMS）？",
    },
    "A-TOY-027": {
        "en": "Has the product been confirmed as a children's product?",
        "zh": "该商品是否已确认为儿童产品？",
    },
    "A-TOY-028": {
        "en": "Which design, process, component, material, color, or factory element changed?",
        "zh": "哪些设计、工艺、零部件、材料、颜色或工厂要素发生了变更？",
    },
    "A-TOY-029": {
        "en": "Has the factory or a component source changed?",
        "zh": "工厂或零部件来源是否发生变更？",
    },
}


def attribute_question_copy(
    attribute_id: Any, lang: str = DEFAULT_LANGUAGE
) -> str | None:
    """Localized customer question for one exact attribute, or ``None``.

    ``None`` means "no approved copy for this attribute" - the caller then falls
    back to approved attribute metadata. Another attribute's question is never
    returned.
    """
    entry = ATTRIBUTE_QUESTIONS.get(str(attribute_id or "").strip())
    if entry is None:
        return None
    language = normalize_language(lang)
    return entry.get(language) or entry.get(DEFAULT_LANGUAGE) or None


def attribute_question_ids() -> set[str]:
    """Every ``attribute_id`` with approved bilingual question copy."""
    return set(ATTRIBUTE_QUESTIONS)


def normalize_language(value: Any) -> str:
    """Return a supported language code, falling back to English."""
    code = str(value or "").strip().lower()
    return code if code in TRANSLATIONS else DEFAULT_LANGUAGE


def t(key: str, lang: str = DEFAULT_LANGUAGE, **kwargs: Any) -> str:
    """Translate ``key`` into ``lang``.

    Missing translations fall back to English; a key missing everywhere returns
    the key itself so presentation never raises. ``kwargs`` are applied only when
    the (possibly fallback) template contains them.
    """
    language = normalize_language(lang)
    table = TRANSLATIONS.get(language, _EN)
    template = table.get(key)
    if template is None:
        template = _EN.get(key)
    if template is None:
        return key
    if not kwargs:
        return template
    try:
        return template.format(**kwargs)
    except (KeyError, IndexError, ValueError):
        return template


def available_languages() -> list[str]:
    """Stable language codes, English first."""
    return [DEFAULT_LANGUAGE] + sorted(
        code for code in TRANSLATIONS if code != DEFAULT_LANGUAGE
    )


def keys_for(lang: str) -> set[str]:
    """Keys defined for one language (used by tests and tooling)."""
    return set(TRANSLATIONS.get(normalize_language(lang), {}))
