import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb

# 读取训练数据和测试数据
train_data = pd.read_csv('train_data.csv', index_col=0)
test_data = pd.read_csv('test_data.csv', index_col=0)

# 预处理（例如：分割特征和标签）
X_train = train_data.drop('Group', axis=1)
y_train = train_data['Group'].apply(lambda x: 1 if x == 'dead' else 0)
X_test = test_data.drop('Group', axis=1)
y_test = test_data['Group'].apply(lambda x: 1 if x == 'dead' else 0)

# 定义模型和设置参数网格
models = {
    'Logistic Regression': LogisticRegression(max_iter=10000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'LightGBM': lgb.LGBMClassifier(random_state=42)
}

param_grids = {
    'Logistic Regression': {'C': [0.1, 1, 10]},
    'Random Forest': {'n_estimators': [100, 200], 'max_depth': [5, 10]},
    'XGBoost': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5, 7]},
    'Decision Tree': {'max_depth': [3, 5, 7, 10], 'min_samples_split': [2, 5, 10]},
    'LightGBM': {'num_leaves': [31, 50, 100], 'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200]}
}

plt.figure(figsize=(5, 5))
model_colors = ["#17becf", "#ff7f0e", "#7f7f7f", "#bcbd22", "#e377c2"]
model_index = 0


# 准备绘制其他指标
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

performance_list = []

# Training and evaluating models
for name, model in models.items():
    print(f"Training {name}...")
    grid_search = GridSearchCV(model, param_grids[name], cv=5, n_jobs=-1, scoring='roc_auc')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test)
    probabilities = best_model.predict_proba(X_test)[:, 1]

    # ROC曲线和AUC
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', color=model_colors[model_index])

    # 计算并保存其他指标
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    accuracy_scores.append((name, accuracy))
    precision_scores.append((name, precision))
    recall_scores.append((name, recall))
    f1_scores.append((name, f1))

    performance_list.append({
        'Model': name,
        'AUC': roc_auc,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    })

    model_index += 1

    if name == 'XGBoost':
        # XGBoost-specific analysis
        best_model.save_model('final_xgboost_model.json')
        explainer = shap.Explainer(best_model)
        shap_values = explainer(X_test)

        # plt.figure(figsize=(8, 8))
        # shap.summary_plot(shap_values, max_display=15, plot_type="bar", show=False)
        # plt.savefig(f'{name}_SHAP_Summary.pdf')
        # plt.close()

        # 计算每个特征的平均SHAP值绝对值
        shap_values_df = pd.DataFrame(shap_values.values, columns=X_test.columns)
        shap_abs_mean = shap_values_df.abs().mean().sort_values(ascending=False)

        # 绘制前15个特征的重要性柱状图
        top_n = 15
        top_features = shap_abs_mean.head(top_n)

        plt.figure(figsize=(6, 5))
        plt.barh(top_features.index, top_features.values, color="#1f77b4")
        plt.xticks(rotation=45, fontsize=14, fontname='Arial')  # 增大x轴刻度标签字体，使用Arial字体
        plt.yticks(rotation=45, fontsize=14, fontname='Arial')  # 增大y轴刻度标签字体，使用Arial字体
        plt.title("Top Feature Importance (SHAP Values)", fontsize=14, fontname='Arial')
        plt.gca().invert_yaxis()  # 倒转Y轴，使得重要性最高的特征在最上方
        plt.tight_layout()

        # 保存为PDF文件
        plt.savefig(f'{name}_SHAP_Summary.pdf')
        plt.close()

        # 绘制并保存SHAP beeswarm图
        plt.figure(figsize=(8, 8))  # 设置高度为宽度的两倍
        shap.plots.beeswarm(shap_values, max_display=15, show=False)  # 绘制SHAP beeswarm图，最多显示前15个特征
        ax = plt.gca()  # 获取当前轴
        for label in ax.get_yticklabels():
            label.set_fontsize(16)
        plt.savefig('XGBoost_SHAP_Beeswarm.pdf')  # 保存为PDF
        plt.close()  # 关闭图形以释放内存

# Saving performance data to CSV
performance_data = pd.DataFrame(performance_list)
performance_data.to_csv('model_performance.csv', index=False)


# 保存所有模型的ROC曲线到一个PDF文件
plt.title('Comparison of ROC Curves at year 5')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.rcParams.update({'font.size': 12})
plt.savefig('All_Models_ROC_Curves.pdf', bbox_inches='tight', pad_inches=0.1)
plt.close()

# 绘制其他指标图表
def plot_metric(metrics, title, filename, color):
    plt.figure(figsize=(5, 5))
    names, values = zip(*metrics)
    bars = plt.bar(names, values, color=color)
    plt.xlabel('')
    plt.xticks(rotation=15)
    plt.ylabel(title)
    plt.title(f'Comparison of Model {title} at year 5')
    plt.ylim([0, 1])
    # 添加数值标签
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f'{value:.3f}', ha='center', va='bottom')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()

plot_metric(accuracy_scores, 'Accuracy', 'All_Models_Accuracy.pdf', model_colors)
plot_metric(precision_scores, 'Precision', 'All_Models_Precision.pdf', model_colors)
plot_metric(recall_scores, 'Recall', 'All_Models_Recall.pdf', model_colors)
plot_metric(f1_scores, 'F1-Score', 'All_Models_F1-Score.pdf', model_colors)
