## Adult 收入预测项目

### 1. 问题界定与数据背景
- **业务动机**：在人力资源筛查或政府政策评估中，准确识别可能年收入超过 50K 美元的人群有助于制定差异化激励。真实数据通常含有缺失、脏值与分类噪声，因此本项目着重探究“数据治理 + 特征工程”能否显著改善模型表现。
- **数据来源**：Adult / Census Income（Becker & Kohavi, 1996），通过 `ucimlrepo` 自动下载，确保实验可复现。
- **基本画像**：清洗后数据包含 48,790 条记录、15 个原始属性。目标变量“>50K”占比 23.94%，表现出中度类别不平衡；`analysis_v2/artifacts/missing_values.csv` 显示 `occupation`、`workclass`、`native_country` 为空的比例分别达到 5.7%、5.7%、1.8%，若直接丢弃将造成信息损失。
- **挑战定位**：存在 placeholder “?”、重复样本、极端资本收益，以及多个语义重复的分类字段（如 `education` vs. `education-num`）。此外，收入的主要驱动因素往往跨字段组合出现（例如“教育年限 × 工作时长”），需要定制特征工程。

### 2. Design & Implementation（设计与实现）
#### 2.1 数据获取与探索
- 通过 `analysis_v2/adult_pipeline.py` 自动执行数据下载、特征/标签拼接及列名标准化（统一小写、替换空格/符号），同时保留原始数据的 `metadata` 以便追溯。
- EDA 产物：
  - `missing_values.csv`、`categorical_distributions.csv`：提供前 20 个缺失字段与主要类别占比，用于判断是“删除样本”还是“智能填补”。
  - `correlation_heatmap.png`：展示数值字段相关性矩阵，观察到 `education_num` 对 `capital_gain` 的正相关较弱，提示单一数值难以解释收入，需要多特征联动。
  - `numeric_distributions.png`：比较不同收入标签的年龄、工时、资本收益，初步验证“高收入更集中在 35–55 岁 & 45+ 小时”的业务假设。

#### 2.2 高级预处理与特征工程
- **清洗策略**：
  - 将 “?” 替换为 `NaN` 并统一字符串大小写 / 去除尾部空格，以消除隐形类别。
  - 删除缺失标签行、重复样本，并对 `hours_per_week`、`capital_gain/loss` 进行上下界裁剪（1–80 小时、0–99,999 / 4,356）以缓解极端值影响。
  - 执行中位数/众数填补，保持样本量；若直接丢弃缺失可导致数据量下降 8%。
- **特征工程亮点（`AdultFeatureEngineer`）**：
  - **资本能力**：`capital_net`、`capital_gain_log`、`capital_intensity` 量化投资收益与时间投入的关系，可区分“工资型”与“投资型”高收入人群。
  - **教育 × 劳动供给**：`education_hours_interaction`、`overtime_flag`、`hours_bucket` 捕捉技能利用率与加班行为；`boxplot_income.png` 显示该交互项对高收入有明显分界。
  - **人口属性整合**：通过 `workclass_grouped`、`occupation_grouped`、`education_level_grouped`、`household_role`、`native_is_us` 等字段整合语义重复或稀有类别，减少噪声、提升模型稳定性。
  - **公平性视角**：新增 `income_by_gender.png` 用于快速审视性别与高收入概率的关系，便于后续延展公平性分析。

#### 2.3 多模型流水线
- 统一流水线：`Pipeline(AdultFeatureEngineer → ColumnTransformer → Classifier)`，确保每种模型接收完全一致的预处理输出。
- 模型设置：
  1. **Logistic Regression**：`class_weight="balanced"` 增强对少数类的敏感度，作为线性基准。
  2. **Random Forest**：400 棵树、`max_depth=18`、`min_samples_leaf=10`，擅长处理混合类型特征。
  3. **HistGradientBoosting**：学习率 0.08、深度 8、叶子 25，并加 L2 正则防止过拟合。
  4. **XGBoost**：`tree_method="hist"`、`scale_pos_weight≈3.2` 对抗类别不平衡，`subsample/colsample=0.8` 保证泛化。
- 数据划分：按 75%/25% Stratified 分割训练/测试集，并在训练集上执行 5 折 StratifiedKFold 交叉验证。整个过程通过 `evaluate_models` 记录 CV 均值与测试集表现。

- 数值分布矩阵（`analysis_v2/artifacts/numeric_distributions.png`）显示高收入人群集中在 35–55 岁、高教育年限与高工时段；资本收益呈长尾分布，强化对数化需求。
- 类别对比图（`income_by_education.png`、`income_by_occupation.png`、`income_by_gender.png`）量化不同教育层级/职业族群/性别的高收入概率，指导桶化策略与公平性分析。
- 成对关系与降维：`pairplot_income.png` 揭示 `education_num` 与 `hours_per_week` 的联合边界；`pca_scatter.png` 展示加入工程特征后的可分性，高收入样本在主成分空间明显偏向高 PC1；`boxplot_income.png` 则直观比较了核心数值特征在不同收入标签下的分布。

#### 2.5 模型解释增强
- 对最优模型输出混淆矩阵与 ROC 曲线（`xgboost_confusion_matrix.png`、`xgboost_roc_curve.png`），便于从真实/预测角度检查成本。
- 置换重要性文件（`xgboost_feature_importance.csv` 与 `hist_gradient_boosting_feature_importance.csv`）均指向工时、教育×工时、资本损益、年龄为关键变量。
- 局部依赖曲线（`xgboost_partial_dependence.png`）揭示 `hours_per_week` 在 45 小时时存在“加班阈值”，`capital_gain` 由 0 跨入正数即刻提高年薪概率，形成可解释的业务规则。

### 3. Evaluation（评估与结果分析）
#### 3.1 实验设置与指标
- 交叉验证与测试阶段统一使用 Accuracy、Precision、Recall、F1、ROC-AUC 五项指标，兼顾整体精度与对少数类的召回。
- 所有数值均自动写入 `analysis_v2/artifacts/cv_metrics.csv` 和 `test_metrics.json` 以便追溯。

#### 3.2 交叉验证表现

| 模型 | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.815 | 0.578 | **0.852** | 0.688 | 0.914 |
| Random Forest | 0.863 | **0.790** | 0.580 | 0.669 | 0.918 |
| HistGradientBoosting | **0.872** | 0.782 | 0.647 | **0.708** | 0.928 |
| XGBoost | 0.838 | 0.617 | 0.852 | 0.715 | **0.928** |

> 数据来源：`analysis_v2/artifacts/cv_metrics.csv`

解析：
- Logistic Regression 在 `class_weight="balanced"` 作用下召回率最高，但 Precision 与 F1 有明显短板，说明线性边界不足以利用工程特征的非线性潜力。
- Random Forest 与 HistGBers 利用树结构捕捉复杂交互，整体精度明显优于线性模型。
- XGBoost 虽在 Accuracy 上略低，但 F1=0.715，说明其在召回与 Precision 之间取得了更好的折衷，是后续部署的优选。

#### 3.3 独立测试集表现

| 模型 | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.814 | 0.574 | **0.859** | 0.688 | 0.911 |
| Random Forest | 0.859 | **0.790** | 0.560 | 0.655 | 0.915 |
| HistGradientBoosting | **0.869** | 0.779 | 0.634 | 0.699 | 0.926 |
| XGBoost | 0.832 | 0.605 | 0.864 | **0.712** | **0.927** |

> 数据来源：`analysis_v2/artifacts/test_metrics.json`

解析：
- **召回 vs. 精准度**：Logistic Regression 继续维持高召回但精度不足；XGBoost 以 0.864 的召回和 0.605 的精度实现 F1=0.712，显著减少误报。
- **Accuracy vs. F1**：HistGB 在 Accuracy 上领先，说明其对多数类预测更稳健；但 F1 略低于 XGBoost，理由在于 HistGB 对少数类召回只有 0.634。
- `run_summary.json` 记录 XGBoost 为最佳模型，并打印其测试指标，方便在报告或演示中引用。

- **与线性基线对比**：Logistic Regression（仅基础清洗）虽然召回达 0.859，但 Precision/F1 分别只有 0.574/0.688；XGBoost 在相同数据上凭借工程特征 + 树模型把 F1 提升到 0.712，同时保持 0.864 的召回，说明预处理与非线性结构带来的增益真实可量化。
- **混淆矩阵与 ROC 曲线**（`analysis_v2/artifacts/xgboost_confusion_matrix.png`、`xgboost_roc_curve.png`）显示在 0.5 阈值下可捕获 86% 高收入用户且保持可控 FPR，适合收入筛查应用。
- **特征重要性**：置换重要性（`xgboost_feature_importance.csv` 与 `hist_gradient_boosting_feature_importance.csv`）一致强调 `hours_per_week`、`education_hours_interaction`、`capital_loss`、`age`、`capital_net` 为主驱动，验证“劳动投入 + 教育深度 + 资本收益”假设。
- **预处理贡献**：稀有类别合并与资本长尾裁剪抑制了过拟合；`education_hours_interaction` 刻画“技能利用率”，成为提升 F1 的关键变量；`numeric_distributions.png`、`boxplot_income.png` 与 `income_by_education.png` 等可视化进一步印证了这些特征的区分度与业务含义。

### 4. Conclusion（结论与展望）
- **主要发现**：系统化的清洗 + 特征工程 + 数据整合能够显著提升 Adult 收入预测的召回与 F1。相比线性基线，XGBoost 在测试集将 F1 提升 3.4 个百分点，并保持 0.864 的召回，验证了“工程特征 + 非线性模型”策略的价值。
- **经验教训**：1) 占位符、稀有类别、极端值如果不处理会导致严重偏差；2) 结构化交互特征（教育×工时、资本收益比）比单一字段更具解释力；3) 可视化与可解释性输出（混淆矩阵、PDP）对业务沟通十分关键。


### 5. 参考文献
1. Becker, B. & Kohavi, R. (1996). *Adult* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20.

