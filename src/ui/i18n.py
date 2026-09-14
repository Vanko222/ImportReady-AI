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
    "update_analysis": "更新分析",
    "missing_none": "本次分析无需补充信息。",
    "input_issues_title": "信息问题",
    "input_issues_note": "部分已提交的值无法使用，请检查并修正。",
    "value_free_text": "请填写内容",
    "value_free_list": "每行一项",
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
}

#: Fixed translation table. ``en`` is the fallback language for every key.
TRANSLATIONS: dict[str, dict[str, str]] = {"en": _EN, "zh": _ZH}


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
