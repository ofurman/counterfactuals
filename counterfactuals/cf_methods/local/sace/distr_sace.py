import numpy as np

from counterfactuals.sace.sace import SACE


def gaussian_matched_vicinity_sampling(x, epsilon, num_points=1):
    return gaussian_vicinity_sampling(x, epsilon, num_points) / np.sqrt(
        1 + (epsilon**2)
    )


def gaussian_vicinity_sampling(x, epsilon, num_points=1):
    return x + (np.random.normal(size=(num_points, x.shape[1])) * epsilon)


def gaussian_global_sampling(x, num_points=1):
    return np.random.normal(size=(num_points, x.shape[1]))


def uniform_sphere_origin(num_points, dimensionality, radius=1):
    """Generate 'num_points' random points in 'dimension' that have uniform probability over the unit ball scaled
    by 'radius' (length of points are in range [0, 'radius']).

    Parameters
    ----------
    num_points : int
        number of points to generate
    dimensionality : int
        dimensionality of each point
    radius : float
        radius of the sphere

    Returns
    -------
    array of shape (num_points, dimensionality)
        sampled points
    """
    # First generate random directions by normalizing the length of a
    # vector of random-normal values (these distribute evenly on ball).
    random_directions = np.random.normal(size=(dimensionality, num_points))
    random_directions /= np.linalg.norm(random_directions, axis=0)
    # Second generate a random radius with probability proportional to
    # the surface area of a ball with a given radius.
    random_radii = np.random.random(num_points) ** (1 / dimensionality)
    # Return the list of random (direction & length) points.
    return radius * (random_directions * random_radii).T


def uniform_sphere_vicinity_sampling(x, num_points=1, radius=1):
    Z = uniform_sphere_origin(num_points, x.shape[1], radius)
    translate(Z, x)
    return Z


def uniform_sphere_scaled_vicinity_sampling(x, num_points=1, threshold=1):
    Z = uniform_sphere_origin(num_points, x.shape[1], radius=1)
    Z *= threshold
    translate(Z, x)
    return Z


def translate(X, center):
    """Translates a origin centered array to a new center

    Parameters
    ----------
    X : array
        data to translate centered in the axis origin
    center : array
        new center point

    Returns
    -------
    None
    """
    for axis in range(center.shape[-1]):
        X[..., axis] += center[..., axis]


def vicinity_sampling(
    x,
    nx,
    n=1000,
    threshold=None,
    kind="gaussian_matched",
    variable_features=None,
    categorical_features_lists=None,
    scaler=None,
    pooler=None,
    verbose=True,
    round_value=None,
):
    if verbose:
        print("\nSampling -->", kind)

    if kind == "gaussian":
        Z = gaussian_vicinity_sampling(nx, threshold, n)
    elif kind == "gaussian_matched":
        Z = gaussian_matched_vicinity_sampling(nx, threshold, n)
    elif kind == "gaussian_global":
        Z = gaussian_global_sampling(nx, n)
    elif kind == "uniform_sphere":
        Z = uniform_sphere_vicinity_sampling(nx, n, threshold)
    elif kind == "uniform_sphere_scaled":
        Z = uniform_sphere_scaled_vicinity_sampling(nx, n, threshold)
    else:
        raise Exception("Vicinity sampling kind not valid")

    if scaler:
        Z = scaler.inverse_transform(Z)

    if categorical_features_lists is not None:
        Z = correct_categorical(Z, categorical_features_lists)

    Z = update_with_unmutable_feature(x, Z, variable_features, pooler, round_value)

    return Z


def correct_categorical(Z, categorical_features_lists):
    # if not categorical_features_lists:
    #     return Z

    for idx_list in categorical_features_lists:
        idx_set_to_one = np.argmax(Z[:, idx_list], axis=1)
        vals = np.zeros((len(Z), len(idx_list)))
        for i, j in zip(range(len(Z)), idx_set_to_one):
            vals[i, j] = 1
        Z[:, idx_list] = vals

    return Z


def update_with_unmutable_feature(
    x, Z, variable_features, pooler=None, round_value=None
):
    Z_updated = list()
    for cf_idx in range(len(Z)):
        cfc = x.copy() if not pooler else pooler.transform(x)
        # cfc[:, variable_features] = Z[cf_idx, variable_features] if round_value is None \
        #     else round_value(Z[cf_idx, variable_features])
        cfc[:, variable_features] = round_value(Z[cf_idx, variable_features])
        # cfc[:, variable_features] = Z[cf_idx, variable_features]
        Z_updated.append(cfc.flatten())

    Z_updated = np.array(Z_updated)

    return Z_updated


def binary_sampling_search(
    x,
    nx,
    label,
    predict,
    lower_threshold=0,
    upper_threshold=4,
    n_attempts=10000,
    n_batch=1000,
    stopping_eps=0.01,
    kind="gaussian_matched",
    downward_only=True,
    variable_features=None,
    categorical_features_lists=None,
    scaler=None,
    pooler=None,
    verbose=True,
    round_value=None,
    max_iter_count=10,
):
    if verbose:
        print("Binary sampling search:")

    # sanity check for the upper threshold
    not_found = True
    for i in range(n_attempts):
        Z = vicinity_sampling(
            x=x,
            nx=nx,
            n=n_batch,
            threshold=upper_threshold,
            kind=kind,
            variable_features=variable_features,
            categorical_features_lists=categorical_features_lists,
            scaler=scaler,
            pooler=pooler,
            verbose=verbose,
            round_value=round_value,
        )
        y = predict(Z)
        if not np.all(y == label):
            not_found = False
            break

    if not_found:
        return None
        # raise Exception('No counterfactual found, increase upper threshold or n_search.')

    change_lower = False
    Z_counterfactuals = list()
    latest_working_threshold = upper_threshold
    threshold = (lower_threshold + upper_threshold) / 2
    # while upper_threshold != 0 and lower_threshold / upper_threshold < stopping_eps:
    iter_count = 0
    while iter_count < max_iter_count:
        # print(iter_count)
        iter_count += 1
        if change_lower:
            if downward_only:
                break
            lower_threshold = threshold

        threshold = (lower_threshold + upper_threshold) / 2
        change_lower = True

        if verbose:
            print(
                "   Testing threshold value:",
                threshold,
                upper_threshold,
                lower_threshold,
            )

        for i in range(n_attempts):
            Z = vicinity_sampling(
                x=x,
                nx=nx,
                n=n_batch,
                threshold=threshold,
                kind=kind,
                variable_features=variable_features,
                categorical_features_lists=categorical_features_lists,
                scaler=scaler,
                pooler=pooler,
                verbose=verbose,
                round_value=round_value,
            )
            y = predict(Z)
            if not np.all(y == label):  # if we found already some counterfactuals
                counterfactuals_idxs = np.argwhere(y != label).ravel()
                Z_counterfactuals.append(Z[counterfactuals_idxs])
                latest_working_threshold = threshold
                upper_threshold = threshold
                change_lower = False
                break

        if np.abs(upper_threshold - lower_threshold) < stopping_eps:
            break

    if verbose:
        print("   Best threshold found:", latest_working_threshold)

    if verbose:
        print("   Final counterfactual search", end=" ")

    Z = vicinity_sampling(
        x=x,
        nx=nx,
        n=n_batch,
        threshold=latest_working_threshold,
        kind=kind,
        variable_features=variable_features,
        categorical_features_lists=categorical_features_lists,
        scaler=scaler,
        pooler=pooler,
        verbose=verbose,
        round_value=round_value,
    )
    y = predict(Z)
    counterfactuals_idxs = np.argwhere(y != label).ravel()
    Z_counterfactuals.append(Z[counterfactuals_idxs])
    if verbose:
        print("Done!")

    Z_counterfactuals = np.concatenate(Z_counterfactuals)

    return Z_counterfactuals


class DistrSACE(SACE):
    def __init__(
        self,
        variable_features=None,
        weights=None,
        metric="euclidean",
        feature_names=None,
        continuous_features=None,
        categorical_features_lists=None,
        normalize=True,
        pooler=None,
        tol=0.01,
        n_attempts=10000,
        n_batch=1000,
        stopping_eps=0.01,
        kind="gaussian_matched",
    ):
        super().__init__(
            variable_features,
            weights,
            metric,
            feature_names,
            continuous_features,
            categorical_features_lists,
            normalize,
            pooler,
            tol,
        )
        self.n_attempts = n_attempts
        self.n_batch = n_batch
        self.stopping_eps = stopping_eps
        self.kind = kind

    def fit(self, b, X):
        super().fit(b, X)

    def get_counterfactuals(
        self,
        x,
        k=5,
        y_desiderd=None,
        constrain_into_ranges=True,
        search_diversity=False,
        lower_threshold=0,
        upper_threshold=4,
    ):
        # x = x.reshape(1, -1)
        verbose = False
        x = np.expand_dims(x, 0)
        nx = (
            self.scaler.transform(x)
            if not self.pooler
            else self.scaler.transform(self.pooler.transform(x))
        )
        y_val = self.b.predict(x)[0]

        Z = binary_sampling_search(
            x,
            nx,
            y_val,
            self._predict,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
            n_attempts=self.n_attempts,
            n_batch=self.n_batch,
            stopping_eps=self.stopping_eps,
            kind=self.kind,
            downward_only=True,
            variable_features=self.variable_features,
            categorical_features_lists=self.categorical_features_index_lists,
            scaler=self.scaler,
            pooler=self.pooler,
            verbose=verbose,
            round_value=self._round_value,
        )

        count_search = 1
        while Z is None:
            count_search += 1
            Z = binary_sampling_search(
                x,
                nx,
                y_val,
                self._predict,
                lower_threshold=lower_threshold,
                upper_threshold=upper_threshold * count_search,
                n_attempts=self.n_attempts,
                n_batch=self.n_batch,
                stopping_eps=self.stopping_eps,
                kind=self.kind,
                downward_only=True,
                variable_features=self.variable_features,
                categorical_features_lists=self.categorical_features_index_lists,
                scaler=self.scaler,
                pooler=self.pooler,
                verbose=verbose,
                round_value=self._round_value,
            )

            if count_search >= 4:
                break

        if Z is None:
            cf_list = np.array([])
            return cf_list

        nZ = self.scaler.transform(Z)  # if not self.pooler else \
        # self.scaler.transform(self.pooler.transform(Z))
        dists = self.cdist(nx, nZ, metric=self.metric, w=self.weights)

        cf_score = dict()
        cf_list_set = set()
        for cf_idx in np.argsort(dists)[0]:
            cfc = Z[cf_idx]
            cfc_tuple = tuple(cfc.flatten())
            if cfc_tuple not in cf_list_set:
                cf_list_set.add(cfc_tuple)
                score = dists[0][cf_idx]
                cf_score[len(cf_score)] = (score, cfc.flatten())
                if len(cf_score) == k:
                    break

        if len(cf_score) > k and search_diversity:
            cf_list = self._get_diverse(cf_score, k)
        else:
            cf_list = self._get_closest(cf_score, k)

        # print(x, 'x')
        # print('----')
        # print(cf_list)

        if self.pooler:
            cf_list = self.pooler.inverse_transform(cf_list)

        return cf_list

    def get_prototypes(
        self, x, k=5, beta=0.5, constrain_into_ranges=True, search_diversity=False
    ):
        x = np.expand_dims(x, 0)
        nx = (
            self.scaler.transform(x)
            if not self.pooler
            else self.scaler.transform(self.pooler.transform(x))
        )
        y_val = self.b.predict(x)[0]
        y_prob = self.b.predict_proba(x)[:, y_val][0]

        cond = self.y == y_val

        X_prc = self.X[cond]
        nX_prc = (
            self.scaler.transform(X_prc)
            if not self.pooler
            else self.scaler.transform(self.pooler.transform(X_prc))
        )

        dists = self.cdist(nx, nX_prc, metric=self.metric, w=self.weights)

        pr_score = dict()
        pr_list_set = set()
        for cf_idx in np.argsort(dists)[0]:
            prc = x.copy() if not self.pooler else self.pooler.transform(x)
            prc[:, self.variable_features] = X_prc[cf_idx, self.variable_features]
            y_prc = self._predict(prc)[0]
            y_prc_prob = self._predict_proba(prc)[:, y_prc][0]

            if y_prc == y_val and y_prc_prob > y_prob:
                if not self._respect_ranges(prc):
                    if constrain_into_ranges:
                        prc = self._contrain_into_ranges(prc)
                    else:
                        continue

                if not self._respect_categorical_features(prc):
                    continue

                prc_tuple = tuple(prc.flatten())
                if prc_tuple not in pr_list_set:
                    pr_list_set.add(prc_tuple)
                    n_prc = self.scaler.transform(prc)
                    score = self._calculate_prototype_score(nx, n_prc, beta=beta)
                    pr_score[len(pr_score)] = (score, prc.flatten())

        if len(pr_score) > k and search_diversity:
            pr_list = self._get_diverse(pr_score, k)
        else:
            pr_list = self._get_closest(pr_score, k)

        if self.pooler:
            pr_list = self.pooler.inverse_transform(pr_list)

        return pr_list
