"""Affine readouts, restricted pairwise comparison, and cross-context transfer."""
import numpy as np
import torch
from scipy.stats import pearsonr, ttest_rel
from sklearn.svm import LinearSVC
from .geometry import p_classes, ablate


def train_affine(x, y, train, validation, *, seed, epochs=200, lr=1e-3, classes=None):
    """Early stopping uses validation only; no scaler/PCA fitted on test images."""
    torch.manual_seed(seed)
    tx = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    ty = torch.as_tensor(y, dtype=torch.long)
    if set(train) & set(validation):raise ValueError("Train/validation leakage")
    if len(train) == 0 or len(validation) == 0:raise ValueError("Empty decoder split")
    classes = int(np.max(y) + 1) if classes is None else int(classes)
    model = torch.nn.Linear(x.shape[1], classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss, best, best_epoch = float("inf"), None, None
    # Full-batch reference implementation makes the objective easy to inspect.
    for epoch in range(epochs):
        model.train();optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(tx[train]), ty[train])
        loss.backward();optimizer.step()
        model.eval()
        with torch.no_grad():v = float(torch.nn.functional.cross_entropy(model(tx[validation]), ty[validation]))
        if v < best_loss:
            best_loss, best_epoch = v, epoch + 1
            best = {k: t.detach().clone() for k, t in model.state_dict().items()}
    if best is None:raise ValueError("Decoder produced no finite validation loss")
    model.load_state_dict(best)
    return model.eval(), {"selected_epoch": best_epoch, "validation_loss": best_loss}


def predict(model, x):
    with torch.no_grad():return model(torch.as_tensor(x, dtype=torch.float32)).argmax(1).numpy()


def decoder_metrics(x, table, bases, *, seed=1729, basis_scope="full", epochs=200, lr=1e-3):
    fold = table.fold3_id.to_numpy()
    train, validation, test = [np.flatnonzero(fold == f) for f in range(3)]
    labels = {"N": table.N.to_numpy(dtype=int) - 1,
              "P": p_classes(table.N.to_numpy(), table.w.to_numpy())}
    # Only observed NP pairs are classes; encode using a stable complete grid map.
    pairs = sorted(set(zip(labels["N"], labels["P"])))
    mapping = {pair: k for k, pair in enumerate(pairs)}
    labels["NP"] = np.array([mapping[p] for p in zip(labels["N"], labels["P"])])
    outputs, models, predictions = {}, {}, {}
    for target, y in labels.items():
        model, selection = train_affine(x, y, train, validation, seed=seed,
                                       epochs=epochs, lr=lr, classes=int(y.max()+1))
        original = predict(model, x[test]);true = y[test]
        result = {"accuracy": float(np.mean(original == true)), "n_test": len(test), **selection}
        if target in ("N", "P"):
            removed = "P" if target == "N" else "N"
            changed = predict(model, ablate(x[test], bases["U_"+removed]))
            baseline = float(np.abs(original - true).mean())
            altered = float(np.abs(changed - true).mean())
            result.update(baseline_ordinal_mae=baseline, ablated_ordinal_mae=altered,
                          cross_quantity_damage=altered-baseline, removed_quantity=removed,
                          removed_rank=bases["U_"+removed].shape[1],
                          ablation_origin="raw_zero", basis_scope=basis_scope,
                          decoder_retrained=False)
            predictions[target+"_ablated"] = changed
        outputs[target] = result;models[target] = model
        predictions[target+"_true"] = true;predictions[target+"_pred"] = original
    predictions["test_indices"] = test
    return outputs, models, predictions


def comparison_svm(x, table, target, *, C=1., max_iter=10000, restrict_gap=True):
    """All eligible unordered adjacent/restricted label pairs; image-wise splits."""
    if target not in ("N", "P"):raise ValueError("Expected N or P")
    n, p = table.N.to_numpy(int), p_classes(table.N.to_numpy(), table.w.to_numpy())
    labels, other = (n, p) if target == "N" else (p, n)
    folds = table.fold3_id.to_numpy()
    records = []
    for a in sorted(set(labels)):
        for b in sorted(set(labels)):
            if b <= a or (restrict_gap and b-a > (2 if target == "N" else 1)):continue
            eligible = np.flatnonzero((labels == a) | (labels == b))
            train = eligible[folds[eligible] == 0];test = eligible[folds[eligible] == 2]
            if set(labels[train]) != {a,b} or set(labels[test]) != {a,b}:continue
            # Learning compares pairs of representations. Both orientations are
            # present so a sign bias cannot solve the task. No image crosses split.
            def pair_examples(ids):
                left, right = ids[labels[ids] == a], ids[labels[ids] == b]
                count = min(len(left),len(right));left,right=left[:count],right[:count]
                differences = x[right] - x[left]
                conflict = other[right] < other[left]
                return np.concatenate([differences,-differences]), np.r_[np.ones(count),np.zeros(count)], np.r_[conflict,conflict]
            train_x, train_y, _ = pair_examples(train)
            test_x, test_y, conflict = pair_examples(test)
            model = LinearSVC(C=C, max_iter=max_iter, dual="auto", random_state=0).fit(train_x,train_y)
            predicted = model.predict(test_x)
            records.append({"a":int(a),"b":int(b),"accuracy":float((predicted==test_y).mean()),
                            "incongruent_accuracy":float((predicted[conflict]==test_y[conflict]).mean()) if conflict.any() else None,
                            "n_test_pairs":len(test_y),"n_incongruent_pairs":int(conflict.sum()),
                            "solver_iterations":int(model.n_iter_), "max_iter":max_iter})
    # Reference matching rule is intentionally explicit, not historical frozen pairs.
    return {"pair_definition":"deterministic sorted-index equal-count matching; both orientations",
            "aggregation":"equal weight over label pairs", "pairs":records,
            "accuracy":float(np.mean([r["accuracy"] for r in records])) if records else None,
            "incongruent_accuracy":float(np.mean([r["incongruent_accuracy"] for r in records if r["incongruent_accuracy"] is not None])) if any(r["incongruent_accuracy"] is not None for r in records) else None}


def context_generalization(x, table, target, *, seed=1729, epochs=200, lr=1e-3):
    """Directional transfer between the lower/upper half of the other quantity."""
    n, p = table.N.to_numpy(int)-1, p_classes(table.N.to_numpy(),table.w.to_numpy())
    y, contexts = (n,p) if target == "N" else (p,n)
    levels = sorted(set(contexts));cut=len(levels)//2
    groups = [set(levels[:cut]), set(levels[cut:])]
    fold = table.fold3_id.to_numpy()
    rows=[]
    for direction,(source,dest) in enumerate([groups,groups[::-1]]):
        seen=np.isin(contexts,list(source));unseen=np.isin(contexts,list(dest))
        train=np.flatnonzero(seen & (fold==0));validation=np.flatnonzero(seen & (fold==1))
        test=np.flatnonzero(unseen & (fold==2))
        common=set(y[train]) & set(y[validation]) & set(y[test])
        if len(common)<2:raise ValueError("Cross-context transfer lacks common target labels")
        train=train[np.isin(y[train],list(common))];validation=validation[np.isin(y[validation],list(common))]
        test=test[np.isin(y[test],list(common))]
        model,selection=train_affine(x,y,train,validation,seed=seed+direction,epochs=epochs,lr=lr)
        predicted=predict(model,x[test])
        rows.append({"train_contexts":sorted(source),"test_contexts":sorted(dest),
                     "common_target_classes":sorted(map(int,common)),"accuracy":float((predicted==y[test]).mean()),
                     "n_test":len(test),**selection})
    return {"directions":rows,"accuracy":float(np.mean([r["accuracy"] for r in rows])),
            "definition":"bidirectional half-context transfer; common label support only; reference split"}


def paired_statistics(before, after):
    """Pair explicit seed -> value mappings, never equal-length positions.

    Unequal cohorts use their shared initialization seeds only; full and
    unmatched cohort sizes/identities remain in the result. This estimates a
    matched-subset difference, not the difference of the two complete cohorts.
    """
    from collections.abc import Mapping
    if not isinstance(before,Mapping) or not isinstance(after,Mapping):
        raise ValueError("Paired tests require explicit initialization-seed mappings")
    def validate(data):
        mapped={int(seed):float(value) for seed,value in data.items()}
        if len(mapped)!=len(data) or not np.isfinite(list(mapped.values())).all():
            raise ValueError("Duplicate seed identity or nonfinite cohort value")
        return mapped
    before,after=validate(before),validate(after)
    shared=sorted(set(before)&set(after))
    if len(shared)<2:raise ValueError("At least two explicitly shared seeds are required")
    b=np.array([before[s] for s in shared]);a=np.array([after[s] for s in shared])
    # Original paired-geometry code used ttest_rel(pretrained, untrained).
    t,p=ttest_rel(a,b)
    return {"n_before":len(before),"n_after":len(after),"n_pairs":len(shared),"df":len(shared)-1,
            "paired_seeds":shared,"unmatched_before_seeds":sorted(set(before)-set(after)),
            "unmatched_after_seeds":sorted(set(after)-set(before)),
            "pairing":"explicit shared initialization seed intersection",
            "t_direction":"after_minus_before","t":float(t),"p_two_sided":float(p),
            "mean_difference_after_minus_before":float(np.mean(a-b)),
            "estimate_scope":"matched seeds only; not the complete-cohort mean difference"}


def state_correlation(x, y):
    x,y=np.asarray(x,float),np.asarray(y,float)
    if len(x)<3 or len(x)!=len(y) or not np.isfinite([x,y]).all():
        raise ValueError("Correlation needs at least three matched finite state means")
    r,p=pearsonr(x,y)
    return {"n_states":len(x),"pearson_r":float(r),"p_two_sided":float(p),
            "replication_unit":"state mean; correlated derived states limit independence"}
