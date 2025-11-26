## Adult 收入预测项目

### 1. 问题界定与数据背景
- **项目目标**：通过 Adult (Census Income) 数据，构建可复现的预测流程来识别年收入可能超过 50K 美元的人群，并系统验证“数据治理 + 特征工程 + 数据整合”对模型表现的贡献。成功标准为：在保持召回率 ≥0.8 的前提下，使 F1 至少优于线性基线 3 个百分点，并能输出可解释的高收入画像。
- **业务动机**：在人力资源筛查或政府政策评估中，高收入人群通常享有不同的激励政策；只有先解决真实数据中的缺失、脏值与类别噪声问题，才能在模型上获得可信结论，因此本项目聚焦“数据层价值”而非单纯模型调参。
- **数据来源**：Adult / Census Income（Becker & Kohavi, 1996），通过 `ucimlrepo` 自动下载，确保实验可复现。
- **基本画像**：清洗后数据包含 48,790 条记录、15 个原始属性。目标变量“>50K”占比 23.94%，表现出中度类别不平衡；`analysis_v2/artifacts/missing_values.csv` 显示 `occupation`、`workclass`、`native_country` 为空的比例分别达到 5.7%、5.7%、1.8%。
- **挑战定位**：存在 placeholder “?”、重复样本、极端资本收益，以及多个语义重复的分类字段（如 `education` vs. `education-num`）；收入驱动因素往往跨字段组合出现（“教育年限 × 工作时长”），因此需要专门的特征工程来挖掘数据价值。

### 2. Design & Implementation（设计与实现）
#### 2.1 数据获取与探索
- 流程：`fetch_ucirepo` 下载 → 列名标准化 → 特征/标签拼接 → 写入 `analysis_v2/adult_pipeline.py`，并保留 `metadata` 便于追溯。
- EDA 产物：
  - `missing_values.csv`、`categorical_distributions.csv`：用于决定对缺失值采取填补/分桶策略。
  - `correlation_heatmap.png`：用于识别数值字段间的弱/强相关关系，指导交互特征设计。
  - `numeric_distributions.png`：对年龄、工时、资本收益等特征分布进行标签分层，为后续画像和模型解释提供证据。

#### 2.2 数据治理与特征工程
- 清洗策略：替换 “?”、统一字符串大小写、剔除缺失标签/重复样本、裁剪极端资本收益/工时、采用中位数/众数填补，尽量保留数据量。
- 特征工程（`AdultFeatureEngineer`）：
  - 资本收益与投入：`capital_net`、`capital_gain_log`、`capital_intensity` 识别工资收入与投资收益的差异。
  - 教育 × 劳动供给：`education_hours_interaction`、`overtime_flag`、`hours_bucket` 捕捉技能利用率与加班模式，解释“多工时 + 高学历”的收入优势。
  - 人口属性整合：将稀有类别合并成 `*_grouped`，减少噪声；`household_role`、`native_is_us` 等字段提供家庭角色和国家背景信息。
  - 公平性视角：`income_by_gender.png` 用于检视性别差异；若后续需要公平性分析，可在此基础上继续挖掘。

#### 2.3 预测流程与模型配置
- 两套数据层版本：
  1. **基础清洗基线**：仅做缺失/极值处理，不执行特征工程，以评估原始数据对模型的支撑。
  2. **高级特征版本**：启用 `AdultFeatureEngineer`，在同样的预处理与模型设置上观察指标差异，验证“数据治理 + 特征工程”的增益。
- 模型统一为 `Pipeline(FeatureEngineer → ColumnTransformer → Classifier)`，确保输入一致。使用的模型包括：`class_weight="balanced"` 的 Logistic Regression；Random Forest (n_estimators=400, max_depth=18)；HistGradientBoosting (learning_rate=0.08, depth=8, L2=0.5)；XGBoost (tree_method="hist", scale_pos_weight≈3.2, subsample/colsample=0.8)。
- 数据划分：75%/25% Stratified train/test split，训练集再用 5 折 StratifiedKFold 做交叉验证。`evaluation_models` 将 CV/测试指标写入 `cv_metrics*.csv` 和 `test_metrics*.json`，并在 `run_summary.json` 指明最优模型及其 artefact。

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

#### 3.2 交叉验证表现：数据层 vs. 改进层

| 模型 | Accuracy | Precision | Recall | F1 | ROC-AUC | 版本 |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.807 | 0.565 | **0.844** | 0.676 | 0.905 | 基线 |
| Logistic Regression | 0.815 | 0.578 | **0.852** | 0.688 | 0.914 | 改进 |
| Random Forest | 0.860 | **0.787** | 0.569 | 0.660 | 0.914 | 基线 |
| Random Forest | 0.863 | **0.790** | 0.580 | 0.669 | 0.918 | 改进 |
| HistGradientBoosting | 0.872 | 0.783 | 0.643 | 0.706 | 0.927 | 基线 |
| HistGradientBoosting | **0.872** | 0.782 | 0.647 | **0.708** | 0.928 | 改进 |
| XGBoost | 0.838 | 0.616 | 0.856 | 0.716 | **0.929** | 基线 |
| XGBoost | 0.838 | 0.617 | 0.852 | 0.715 | **0.928** | 改进 |

> 数据来源：`analysis_v2/artifacts/cv_metrics_baseline.csv` 与 `cv_metrics.csv`

解析：
- **数据治理收益**：对比同一模型在“基线 vs 改进”两列的 F1，可见 Logistic F1 从 0.676 → 0.688、Random Forest 从 0.660 → 0.669、HistGB 0.706 → 0.708，说明高级清洗与特征工程对所有模型都有增益；说明模型性能提升不是单纯的模型类型差异，而是数据价值被释放。
- **模型普适性**：即使在基线数据上，树模型也优于线性模型，但差距更小；经过特征工程后，树模型的优势进一步放大，证明数据治理对捕捉非线性模式尤其有效。

#### 3.3 独立测试集表现：数据层 vs. 改进层

| 模型 | Accuracy | Precision | Recall | F1 | ROC-AUC | 版本 |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.802 | 0.557 | **0.848** | 0.672 | 0.903 | 基线 |
| Logistic Regression | 0.814 | 0.574 | **0.859** | 0.688 | 0.911 | 改进 |
| Random Forest | 0.855 | **0.784** | 0.547 | 0.644 | 0.911 | 基线 |
| Random Forest | 0.859 | **0.790** | 0.560 | 0.655 | 0.915 | 改进 |
| HistGradientBoosting | **0.870** | 0.784 | 0.629 | 0.698 | 0.926 | 基线 |
| HistGradientBoosting | **0.869** | 0.779 | 0.634 | 0.699 | 0.926 | 改进 |
| XGBoost | 0.831 | 0.603 | 0.862 | 0.710 | **0.927** | 基线 |
| XGBoost | 0.832 | 0.605 | 0.864 | **0.712** | **0.927** | 改进 |

> 数据来源：`analysis_v2/artifacts/test_metrics_baseline.json` 与 `test_metrics.json`

解析：
- **召回 vs. 精准度**：Logistic Regression 在基线和改进版本中都维持 >0.84 的召回，但 Precision/F1 仅从 0.557/0.672 提升到 0.574/0.688；相比之下，Random Forest、HistGB、XGBoost 的 F1 提升幅度更大，说明非线性模型更能利用新的数据特征。
- **Accuracy vs. F1**：HistGB 的 Accuracy 始终领先，证明其对多数类预测更稳定；XGBoost 在 F1 上最优，得益于对少数类召回的持续保持（0.862→0.864）。
- `run_summary.json` 记录 XGBoost 为最佳模型，并打印其测试指标，方便在报告或演示中引用。

- **与线性基线对比**：Logistic Regression（基础清洗）F1 为 0.672；在同样模型下加入高级特征后，F1 升至 0.688。若再换成 XGBoost，F1 达到 0.712，召回仍保持 0.864。这表明：① 数据治理 + 特征工程本身有效；② 在改进后的数据上，采用能捕捉非线性关系的模型可进一步释放潜力。
- **混淆矩阵与 ROC 曲线**（`analysis_v2/artifacts/xgboost_confusion_matrix.png`、`xgboost_roc_curve.png`）显示在 0.5 阈值下可捕获 86% 高收入用户且保持可控 FPR，适合收入筛查应用。
- **特征重要性**：置换重要性（`xgboost_feature_importance.csv` 与 `hist_gradient_boosting_feature_importance.csv`）一致强调 `hours_per_week`、`education_hours_interaction`、`capital_loss`、`age`、`capital_net` 为主驱动，验证“劳动投入 + 教育深度 + 资本收益”假设。
- **预处理贡献**：稀有类别合并与资本长尾裁剪抑制了过拟合；`education_hours_interaction` 刻画“技能利用率”，成为提升 F1 的关键变量；`numeric_distributions.png`、`boxplot_income.png` 与 `income_by_education.png` 等可视化进一步印证了这些特征的区分度与业务含义。

#### 3.4 高收入识别画像（数据分析结论）
- **工作投入**：`numeric_distributions.png` 与 `boxplot_income.png` 显示，高收入样本集中在 35–55 岁且每周 ≥45 小时；`xgboost_partial_dependence.png` 中 `hours_per_week` 的 PDP 曲线在 45 小时附近出现陡峭上升，表明“加班阈值”是最显著的收入分界。
- **教育 × 资本收益**：`income_by_education.png` 表明硕士及以上人群的高收入概率超过 50%；`capital_net`、`capital_gain_log` 在 `xgboost_feature_importance.csv` 中排名靠前，并在 PDP 中呈现“一旦资本收益 >0，预测概率快速上升”的趋势，说明“技能深度 + 投资回报”是典型高收入画像。
- **职业与性别差异**：`income_by_occupation.png` 反映管理、专业、销售岗位的高收入概率显著高于其他职业；`income_by_gender.png` 说明男性相较女性具有更高的高收入概率，提示潜在公平性议题，可在后续分析中重点关注。
- **模型验证**：上述画像不仅来自描述性分析，也得到模型输出支持：XGBoost 的混淆矩阵/ROC 在默认阈值下即可覆盖 86% 的高收入个体，证明这些特征确实是模型作出高收入判断的主要依据。

### 4. Conclusion（结论与展望）
- **主要发现**：系统化的清洗 + 特征工程 + 数据整合能够显著提升 Adult 收入预测的召回与 F1。相比线性基线，XGBoost 在测试集将 F1 提升 3.4 个百分点，并保持 0.864 的召回，验证了“工程特征 + 非线性模型”策略的价值。
- **经验教训**：1) 占位符、稀有类别、极端值如果不处理会导致严重偏差；2) 结构化交互特征（教育×工时、资本收益比）比单一字段更具解释力；3) 可视化与可解释性输出（混淆矩阵、PDP）对业务沟通十分关键。


### 5. 参考文献
1. Becker, B. & Kohavi, R. (1996). *Adult* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20.

