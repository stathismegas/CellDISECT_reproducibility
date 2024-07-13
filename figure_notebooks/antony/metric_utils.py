import numpy as np
import scib
import scanpy as sc
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
    
def asw_report(model, adata, cats, cats_to_check, label_key):
    asw_results = {}
    bio_results = []
    # batch_resutls = []
    # Z_0
    adata.obsm[f'dis2p_cE_Z_0'] = model.get_latent_representation(nullify_cat_covs_indices=[s for s in range(len(cats))], nullify_shared=False)

    for i in range(len(cats)):
        if cats[i] not in cats_to_check:
            continue
        null_idx = [s for s in range(len(cats)) if s != i]
        # Z_i
        adata.obsm[f'dis2p_cE_Z_{i+1}'] = model.get_latent_representation(nullify_cat_covs_indices=null_idx, nullify_shared=True)

    bio = scib.metrics.silhouette(adata, label_key, f'dis2p_cE_Z_0', metric='euclidean', scale=True)
    bio_results.append(bio)
    for i in range(len(cats)):
        if cats[i] not in cats_to_check:
            continue
        bio = scib.metrics.silhouette(adata, label_key, f'dis2p_cE_Z_{i+1}', metric='euclidean', scale=True)
        bio_results.append(bio)
        # for j in range(len(cats)):
        #     if j == i:
        #         continue
        #     label_key = cats[j]
        #     batch = scib.metrics.silhouette(adata, label_key, f'dis2p_cE_Z_{i+1}', metric='euclidean', scale=True)
        #     batch_resutls.append(batch)
            
    asw_results['ASW_bio'] = np.mean(bio_results)
    # asw_results['ASW_batch'] = np.mean(batch_resutls)
    return asw_results


def knn_purity(model, adata, label_key, cats, n_neighbors=30):
    """Computes KNN Purity metric for ``adata`` given the batch column name.
        Parameters
        ----------
        adata: :class:`~anndata.AnnData`
            Annotated dataset.
        label_key: str
            Name of the column which contains information about different studies in ``adata.obs`` data frame.
        n_neighbors: int
            Number of nearest neighbors.
        Returns
        -------
        score: float
            KNN purity score. A float between 0 and 1.
    """
    i = cats.index(label_key)
    null_idx = [s for s in range(len(cats)) if s != i]
    # Z_i
    latent_name = f'dis2p_cE_Z_{i+1}'
    adata.obsm[latent_name] = model.get_latent_representation(nullify_cat_covs_indices=null_idx, nullify_shared=True)
    if sparse.issparse(adata.obsm[latent_name]):
        adata.obsm[latent_name] = adata.obsm[latent_name].A
    
    labels = LabelEncoder().fit_transform(adata.obs[label_key].to_numpy())

    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(adata.obsm[latent_name])
    indices = nbrs.kneighbors(adata.obsm[latent_name], return_distance=False)[:, 1:]
    neighbors_labels = np.vectorize(lambda i: labels[i])(indices)

    # pre cell purity scores
    scores = ((neighbors_labels - labels.reshape(-1, 1)) == 0).mean(axis=1)
    res = [
        np.mean(scores[labels == i]) for i in np.unique(labels)
    ]  # per cell-type purity

    return np.mean(res)

