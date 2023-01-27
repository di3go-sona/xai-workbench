# %%
import pandas as pd
import numpy as np
import sklearn.preprocessing

# %%
raw_houses_df = pd.read_csv('../data/01_houses/train.csv').dropna(axis=1).fillna(0)
houses_df = raw_houses_df.filter(c for c in raw_houses_df.columns if raw_houses_df.dtypes[c] in [np.int64, np.float64])
houses_df = houses_df.drop([ 'BsmtHalfBath' , 'BsmtFullBath', 'Id', 'MSSubClass', 'MiscVal', 'BsmtUnfSF', '3SsnPorch', 'MiscVal', 'YrSold', "MoSold"], axis=1)

# %%
from sklearn.model_selection import train_test_split

y_cols = ['SalePrice']
X_cols = [c for c in houses_df.columns if c not in y_cols]

X, y = houses_df.filter(X_cols), houses_df.filter(y_cols)

X_train, X_test, y_train, y_test = train_test_split(X.astype(np.float32),y, test_size=0.2)

# %%
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


scaler = StandardScaler().fit( X)
regressor = DecisionTreeRegressor()
# clf = regressor.fit(scaler.transform(X_train), y_train/1000)

from sklearn.pipeline import Pipeline

pipeline = make_pipeline(scaler, regressor)
pipeline.fit(X_train, y_train)

# %%
import matplotlib.pyplot as plt
houses_df['PredictedSalePrice'] = pipeline.predict(X)

sorted_houses_df = houses_df.sort_values('SalePrice')[['SalePrice', 'PredictedSalePrice']]
plt.figure()
plt.plot(sorted_houses_df['PredictedSalePrice'].tolist(), alpha=0.5, c='red')
plt.plot(sorted_houses_df['SalePrice'].tolist(), alpha=0.9)
plt.legend(['Predicted Sale Price', 'Real Sale Price' ])
ax = plt.gca()
ax.get_xaxis().set_visible(False)

plt.figure()
sorted_houses_df = houses_df.sort_values('PredictedSalePrice')[['SalePrice', 'PredictedSalePrice']]
plt.plot(sorted_houses_df['SalePrice'].tolist(), alpha=0.5)
plt.plot(sorted_houses_df['PredictedSalePrice'].tolist(), alpha=1, c='red')

plt.legend(['Predicted Sale Price', 'Real Sale Price' ][::-1])
ax = plt.gca()
ax.get_xaxis().set_visible(False)
# plt.plot(sorted(houses_df[['SalePrice', 'PredictedSalePrice']].tolist()))

# %%
import shap

# explainer = shap.explainers.Tree(pipeline, X_train)
# svt = explainer(X_train)
explainer = shap.explainers.Permutation(pipeline.predict, X_train)
svp = explainer(X_train)
# explainer = shap.explainers.Sampling(clf.predict, X_train)
# svs = explainer(X_train)


# %%
# exp_t= shap.Explanation( svt.values, svt.base_values[0], (svt.data), feature_names=X_cols)
exp_p = shap.Explanation( svp.values, svp.base_values[0], (svp.data), feature_names=X_cols)
# exp_s = shap.Explanation( svs.values, svs.base_values, X_scaler.inverse_transform(svs.data), feature_names=X_cols)

# %%
shap.plots.waterfall(exp_p[1])

# %%
shap.plots.waterfall(exp_p[11])

# %%
shap.plots.beeswarm(exp_p) #waterfall(exp_t[10])

# %%
# pd.DataFrame(X_scaler.inverse_transform(X_train) ,columns=X_cols)[['OverallQual', 'GrLivArea', 'BsmtFinSF1', 'TotalBsmtSF', 'YearRemodAdd', 'YearBuilt']]

# %%
# corr = r_regression(X_train, y_train[:,0])
# shap_tree = np.mean(svt.values, 0)
shap_permutation = np.mean(svp.values, 0)
# shap_sampling = np.mean(svs.values, 0)

# %%
# df_pearson = pd.DataFrame([X_cols, ['Pearson']*len(X_cols), (corr).tolist(),  ], index=['feature', 'estimator', 'value'])
# df_shapt = pd.DataFrame([X_cols, ['SHAP - tree']*len(X_cols), (shap_tree).tolist(),  ], index=['feature', 'estimator', 'value'])
# df_shapp = pd.DataFrame([X_cols, ['SHAP - permute']*len(X_cols), (shap_permutation).tolist(),  ], index=['feature', 'estimator', 'value'])
# df_shaps = pd.DataFrame([X_cols, ['SHAP - sample']*len(X_cols), (shap_sampling).tolist(),  ], index=['feature', 'estimator', 'value'])

# df_pearson.loc['value'] = (df_pearson.loc['value'] ) / df_pearson.loc['value'].std() + 1
# df_shapt.loc['value'] = (df_shapt.loc['value'] ) / df_shapt.loc['value'].std() + 1
# df_shapp.loc['value'] = (df_shapp.loc['value'] ) / df_shapp.loc['value'].std() + 1
# df_shaps.loc['value'] = (df_shaps.loc['value'] ) / df_shaps.loc['value'].std() + 1

# df = pd.concat(( df_shapt, df_shapp, ), axis=1).T #df_shaps
# df

# %%
# import matplotlib.pyplot as plt 
# import seaborn as sns

# plt.figure(figsize=(20,20))
# sns.catplot(
#     data=df[(df["feature"] != '2ndFlrSF') & (df["feature"] != 'Id')], kind="bar",
#     y="feature", x="value", hue="estimator",
#      alpha=.6, height=20
# )

# %%
from minisom import MiniSom
import torch



data = np.hstack([X,y])

som = MiniSom(15, 15, data.shape[1], sigma=1.5, learning_rate=.7, activation_distance='euclidean',
              topology='hexagonal', neighborhood_function='gaussian', random_seed=10)

som.train( sklearn.preprocessing.StandardScaler().fit_transform( data), 1000, verbose=True)
data.shape


# %%
xx, yy = som.get_euclidean_coordinates()
umatrix = som.distance_map()
weights = som.get_weights()
weights.shape

# %%
# import sklearn.decomposition
# import seaborn as sns

# def update(n_clusters, X_train):
#     pca = sklearn.decomposition.PCA(n_clusters)
#     xx, yy = pca.fit_transform(scaler.transform(X_train)).T
#     plot = sns.scatterplot(x=xx, y=yy, hue=y_train['SalePrice'])
#     return plot

# update(2, X_train)

# %%
import sklearn.cluster
import sklearn.decomposition



kmeans = sklearn.cluster.KMeans(20)
pca = sklearn.decomposition.PCA(10)

train_shap = pd.DataFrame(svp.values, columns=X_cols)

train_shap['Clusters'] = np.argmin(kmeans.fit_transform(svp.values), axis=1)

clust_train_shap = train_shap.groupby('Clusters').mean().astype('int')

pca_clust_train_shap = pca.fit_transform(clust_train_shap)
pca_clust_train_shap = pd.DataFrame(pca_clust_train_shap, index=[f'cluster {i}' for i in range(pca_clust_train_shap.shape[0])], columns=[f'feature {i}' for i in range(pca_clust_train_shap.shape[1])])


# %%
# def colormap(x):
#     return f"background-color: #%02x%02x%02x ; color: %s"  % tuple([int(c*255) for c in plt.cm.RdBu_r(x/30000+0.5)[:3]]+['black' if abs(x)<15000 else 'white'] )
# train_shap.style.applymap(colormap )


# %%
def colormap(x):
    return f"background-color: #%02x%02x%02x ; color: %s"  % tuple([int(c*255) for c in plt.cm.RdBu_r(x/30000+0.5)[:3]]+['black' if abs(x)<15000 else 'white'] )
clust_train_shap.style.applymap(colormap )


# %%

def colormap(x):
    return f"background-color: #%02x%02x%02x ; color: %s"  % tuple([int(c*255) for c in plt.cm.RdBu_r(x/30000+0.5)[:3]]+['black' if abs(x)<15000 else 'white'] )

s_clust_train_shap = clust_train_shap.reindex((-np.abs(clust_train_shap).sum()).sort_values(axis=0).index, axis=1)
s_clust_train_shap = s_clust_train_shap.reindex((-np.abs(clust_train_shap).sum(axis=1)).sort_values().index, axis=0)

s_clust_train_shap.style.applymap(colormap )


# %%
def colormap(x):
    return f"background-color: #%02x%02x%02x ; color: %s"  % tuple([int(c*255) for c in plt.cm.RdBu_r(x/60000+0.5)[:3]]+['black' if abs(x)<30000 else 'white'] )
pca_clust_train_shap.style.applymap(colormap )


# %%
columns_mapping = {
'LotArea': '🏞',
'OverallQual': '✨',
'OverallCond': '👍',
'YearBuilt': '📅',
'YearRemodAdd': '🧱',
'BsmtFinSF1': '🔽',
'BsmtFinSF2': '⏬',
'TotalBsmtSF': '⬇️',
'1stFlrSF': '1️⃣',
'2ndFlrSF': '2️⃣',
'LowQualFinSF': '',
'GrLivArea': '🌳',
'FullBath': '🛀',
'HalfBath': '🚽',
'BedroomAbvGr': '🛏',
'KitchenAbvGr': '🔪',
'TotRmsAbvGrd': '🔢',
'Fireplaces': '🔥',
'GarageCars': '🏎',
'GarageArea': '🚗',
'WoodDeckSF': '',
'OpenPorchSF': '⛩',
'EnclosedPorch': '🪟',
'ScreenPorch': '🪟',
'PoolArea': '🏊',
}



new_cols = []
for component in pca.components_:
    sorting_index = np.argsort(-np.abs(component), axis=0)
    top_indexes = [sorting_index[i]  for i in range(3)]
    top_cols =[X_cols[i]  for i in top_indexes ]
    top_values =[component[i]  for i in top_indexes ]
    # print(component.shape)
    # top_cols_repr = '\n'.join([columns_mapping.get(v)+v for v in top_cols])
    top_cols_repr = [columns_mapping.get(c, '')+f' {c} {v:.2f}' for i,c,v in zip(top_indexes, top_cols, top_values)]

    new_cols.append(top_cols_repr)


def colormap(x):
    return f"background-color: #%02x%02x%02x ; color: %s"  % tuple([int(c*255) for c in plt.cm.RdBu_r(x/60000+0.5)[:3]]+['black' if abs(x)<30000 else 'white'] )

pca_clust_train_shap.columns = pd.MultiIndex.from_arrays(np.transpose(new_cols))
pca_clust_train_shap_tmp = pca_clust_train_shap.copy()
pca_clust_train_shap_tmp['Price'] = y_train

# train_shap = train_shap['']
pca_clust_train_shap_tmp.style.applymap(colormap )


# %%
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm, colorbar
from matplotlib.lines import Line2D

import matplotlib, mplcairo
print('Default backend: ' + matplotlib.get_backend()) 
matplotlib.use("module://mplcairo.macosx")
print('Backend is now ' + matplotlib.get_backend())

from matplotlib.font_manager import FontProperties
prop = FontProperties(fname='/System/Library/Fonts/Apple Color Emoji.ttc')


def plot_feature(feature_id):
        f = plt.figure(figsize=(6,6))
        ax = plt.gca()
        plt.ioff()

        for i in range(weights.shape[0]):
                for j in range(weights.shape[1]):
                        wy = yy[(i, j)] * np.sqrt(3) / 2
                        hex = RegularPolygon((xx[(i, j)], wy), 
                                        numVertices=6, 
                                        radius=.95 / np.sqrt(3),
                                        facecolor=cm.viridis( weights[i, j, feature_id] / (weights[:,:,feature_id].max() - weights[:,:,feature_id].min())), 
                                        alpha=.8, 
                                        
                                        edgecolor='gray')
                        ax.add_patch(hex)
                        top_col = np.argmax(weights[i,j,:-1], axis=0)
                        col_name =  columns_mapping.get( X_cols[top_col], '')
                        # print(col_weights)
                        ax.text(xx[(i, j)]-0.2, wy-0.2,col_name,fontproperties=prop,)

        # markers = ['o', '+', 'x']
        for cnt, x in enumerate(data):
                w = som.winner(x)

                wx, wy = som.convert_map_to_euclidean(w) 
                wy = wy * np.sqrt(3) / 2
                plt.plot(
                        #  markers[t[cnt]-1], 
                        markerfacecolor='None',
                        markersize=12, 
                        markeredgewidth=2)
                # plt.text(wx, wy, ' a')

        xrange = np.arange(weights.shape[0])
        yrange = np.arange(weights.shape[1])


        divider = make_axes_locatable(plt.gca())
        ax_cb = divider.new_horizontal(size="5%", pad=0.05)    
        cb1 = colorbar.ColorbarBase(ax_cb, cmap='viridis', 
                                orientation='vertical', alpha=.4)

        plt.gcf().add_axes(ax_cb)

        legend_elements = []
        ax.legend(handles=legend_elements, loc='upper left', 
                borderaxespad=0., ncol=3, fontsize=14)

        ax.set_title(houses_df.columns[feature_id])
        return f
plot_feature(-1)

# %% [markdown]
# ### Counterfactuals

# %%
import dice_ml
from dice_ml import Dice

d_iris = dice_ml.Data(dataframe=pd.DataFrame(data, columns=X_cols + y_cols),
                      continuous_features=X_cols,
                      outcome_name="SalePrice")


m_iris = dice_ml.Model(model=pipeline, backend="sklearn", model_type='regressor')
exp_genetic_iris = Dice(d_iris, m_iris, method="genetic")


# %%
query_instances_iris = X_test[2:3]
genetic_iris = exp_genetic_iris.generate_counterfactuals(query_instances_iris, total_CFs=15, desired_range=[150000,160000])
genetic_iris.visualize_as_dataframe()

# %% [markdown]
# ### Explainer Dashboard

# %%
from explainerdashboard import RegressionExplainer, ExplainerDashboard
explainer = RegressionExplainer(pipeline, X_test, y_test)
ExplainerDashboard(explainer).run()



