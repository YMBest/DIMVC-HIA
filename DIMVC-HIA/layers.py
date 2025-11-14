import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_hidden_layers=3):
        super(EnergyNetwork, self).__init__()

        layers = []
        current_dim = input_dim

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, 1))
        layers.append(nn.Softplus())

        self.energy_network = nn.Sequential(*layers)

    def forward(self, x):
        energy = self.energy_network(x).squeeze(-1)  # [B]
        return energy

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, feature_dim, dims):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential()
        for i in range(len(dims)+1):
            if i == 0:
                self.encoder.add_module('Linear%d' % i,  nn.Linear(input_dim, dims[i]))
            elif i == len(dims):
                self.encoder.add_module('Linear%d' % i, nn.Linear(dims[i-1], feature_dim))
            else:
                self.encoder.add_module('Linear%d' % i, nn.Linear(dims[i-1], dims[i]))
            self.encoder.add_module('relu%d' % i, nn.ReLU())

    def forward(self, x):
        return self.encoder(x)


class AutoDecoder(nn.Module):
    def __init__(self, input_dim, feature_dim, dims):
        super(AutoDecoder, self).__init__()
        self.decoder = nn.Sequential()
        dims = list(reversed(dims))
        for i in range(len(dims)+1):
            if i == 0:
                self.decoder.add_module('Linear%d' % i,  nn.Linear(feature_dim, dims[i]))
            elif i == len(dims):
                self.decoder.add_module('Linear%d' % i, nn.Linear(dims[i-1], input_dim))
            else:
                self.decoder.add_module('Linear%d' % i, nn.Linear(dims[i-1], dims[i]))
            self.decoder.add_module('relu%d' % i, nn.ReLU())

    def forward(self, x):
        return self.decoder(x)


class DIMVCHIANetwork(nn.Module):
    def __init__(self, num_views, input_sizes, dims, dim_high_feature, dim_low_feature, num_clusters):
        super(DIMVCHIANetwork, self).__init__()
        self.encoders = list()
        self.decoders = list()
        self.num_views = num_views
        self.num_clusters = num_clusters
        for idx in range(num_views):
            self.encoders.append(AutoEncoder(input_sizes[idx], dim_high_feature, dims))
            self.decoders.append(AutoDecoder(input_sizes[idx], dim_high_feature, dims))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

        self.label_learning_module = nn.Sequential(
            nn.Linear(dim_high_feature, dim_low_feature),
            nn.Linear(dim_low_feature, num_clusters),
            nn.Softmax(dim=1)
        )

        self.masks = {}

        self.ebm_list = nn.ModuleList([
            EnergyNetwork(dim_high_feature, 256, 3)
            for _ in range(num_clusters)
        ])

    def forward(self, data_views):
        lbps = list()
        dvs = list()
        features = list()
        all_masks = []
        q_list = []
        view_sims = []
        dvs_new = list()

        num_views = len(data_views)
        for idx in range(self.num_views):
            data_view = data_views[idx]

            mask = (data_view != 0).any(dim=1)
            all_masks.append(mask)

            high_features = self.encoders[idx](data_view)
            label_probs = self.label_learning_module(high_features)
            data_view_recon = self.decoders[idx](high_features)
            features.append(high_features)
            lbps.append(label_probs)

            q = label_probs
            q_list.append(q)
            dvs.append(data_view_recon)

        sum_probs = torch.zeros_like(lbps[0])
        count_valid = torch.zeros_like(lbps[0])

        for idx in range(self.num_views):
            expanded_mask = all_masks[idx].unsqueeze(1)
            sum_probs += q_list[idx] * expanded_mask.float()
            count_valid += expanded_mask.float()


        for i in range(self.num_views):
            for j in range(i + 1, self.num_views):
                q_i = q_list[i]
                q_j = q_list[j]

                view_sim = self.compute_label_similarity(q_i, q_j, all_masks[i], all_masks[j],
                                                         temperature_l=0.1, normalized=False)
                view_sims.append(view_sim)

        imputed_probs_list = []
        imputed_features_list = []
        view_sim_matrix = torch.zeros((self.num_views, self.num_views), device=q_list[0].device)

        sim_idx = 0
        for i in range(self.num_views):
            for j in range(i + 1, self.num_views):
                view_sim_matrix[i, j] = view_sims[sim_idx]
                view_sim_matrix[j, i] = view_sims[sim_idx]
                sim_idx += 1

        for i in range(self.num_views):
            view_sim_matrix[i, i] = 0.0

        for i in range(self.num_views):
            current_probs = q_list[i].clone()
            current_features = features[i].clone()
            current_mask = all_masks[i]

            missing_mask = ~current_mask
            if not torch.any(missing_mask):
                imputed_probs_list.append(current_probs)
                imputed_features_list.append(current_features)
                continue

            impute_info = {}

            other_views_indices = torch.argsort(view_sim_matrix[i], descending=True)

            for idx in other_views_indices:
                other_mask = all_masks[idx]
                valid_positions = missing_mask & other_mask
                if torch.any(valid_positions):
                    for pos in torch.where(valid_positions)[0]:
                        _, cluster_id = torch.max(lbps[idx][pos], dim=0)
                        impute_info[pos.item()] = cluster_id.item()

                    current_probs[valid_positions] = q_list[idx][valid_positions]
                    missing_mask = missing_mask & ~valid_positions

                if not torch.any(missing_mask):
                    break

            imputed_probs_list.append(current_probs)

            for pos, cluster_id in impute_info.items():
                _, current_clusters = torch.max(current_probs, dim=1)
                same_cluster_mask = (current_clusters == cluster_id) & current_mask

                if torch.sum(same_cluster_mask) > 0:
                    cluster_features = current_features[same_cluster_mask]
                    avg_features = torch.mean(cluster_features, dim=0)
                    current_features[pos] = avg_features

            imputed_features_list.append(current_features)

        probs_list_new = imputed_probs_list
        features_list_new = imputed_features_list

        for idx in range(self.num_views):
            data_view_recon_new = self.decoders[idx](features_list_new[idx])
            dvs_new.append(data_view_recon_new)

        ebm_loss = self.compute_cluster_consistency_loss(features_list_new, probs_list_new)

        return probs_list_new, dvs_new, features, ebm_loss, features_list_new, view_sims

    def compute_label_similarity(self, q_i, q_j, mask_i, mask_j, temperature_l, normalized=False):
        """
        Compute label similarity loss using Focal Loss to reduce the contribution of easy samples
        and increase the weight of hard-to-classify samples.
        """
        mask_i = torch.tensor(mask_i, dtype=torch.bool, device=q_i.device)
        mask_j = torch.tensor(mask_j, dtype=torch.bool, device=q_j.device)

        common_mask = mask_i & mask_j
        if not torch.any(common_mask):
            return torch.tensor(0.0, device=q_i.device)

        q_i_valid = q_i[common_mask]
        q_j_valid = q_j[common_mask]

        q_i = self.target_distribution(q_i_valid)
        q_j = self.target_distribution(q_j_valid)

        q_i = q_i.t()
        q_j = q_j.t()
        N = 2 * self.num_clusters
        q = torch.cat((q_i, q_j), dim=0)

        if normalized:
            sim = (self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / temperature_l).to(q.device)
        else:
            sim = (torch.matmul(q, q.T) / temperature_l).to(q.device)

        sim_i_j = torch.diag(sim, self.num_clusters)
        sim_j_i = torch.diag(sim, -self.num_clusters)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples2(N).to(q_i.device)

        pred_labels = torch.argmax(q, dim=1)
        label_mask = pred_labels.unsqueeze(0) != pred_labels.unsqueeze(1)
        final_mask = mask & label_mask

        neg_values = sim[final_mask]
        neg_count = neg_values.shape[0]

        remainder = neg_count % N
        if remainder != 0:
            neg_count_adjusted = neg_count - remainder
            neg_values = neg_values[:neg_count_adjusted]

        negative_clusters = neg_values.reshape(N, -1)
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        probs = F.softmax(logits, dim=1)[:, 0]
        view_sim = probs.sum()

        view_sim_avg = view_sim / (torch.sum(common_mask).item())
        return view_sim_avg

    def target_distribution(self, q):
        """Compute target distribution"""
        weight = (q ** 2.0) / torch.sum(q, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def mask_correlated_samples2(self, N):
        """Create contrastive learning mask, return cached version if already computed"""
        if N in self.masks:
            return self.masks[N]

        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N // 2):
            mask[i, N // 2 + i] = 0
            mask[N // 2 + i, i] = 0
        mask = mask.bool()

        self.masks[N] = mask
        return mask

    def compute_cluster_consistency_loss(self, features_list, probs_list):
        total_loss = 0.0
        feature_size = features_list[0].size(0)
        device = features_list[0].device
        all_assignments = []

        for view_idx in range(self.num_views):
            _, assignments = torch.max(probs_list[view_idx], dim=1)  # [B]
            all_assignments.append(assignments)

        for cluster_id in range(self.num_clusters):
            all_cluster_features = []
            feature_indices = []

            for sample_idx in range(feature_size):
                for view_idx in range(self.num_views):
                    if all_assignments[view_idx][sample_idx] == cluster_id:
                        all_cluster_features.append(features_list[view_idx][sample_idx])
                        feature_indices.append((view_idx, sample_idx))

            if len(all_cluster_features) > 0:
                all_energies = []
                for view_idx, sample_idx in feature_indices:
                    sample_feature = features_list[view_idx][sample_idx].unsqueeze(0)  # [1, feature_dim]
                    sample_energy = self.ebm_list[cluster_id](sample_feature).squeeze()
                    all_energies.append(sample_energy)

                all_energies_tensor = torch.stack(all_energies, dim=0)  # [n]
                min_energy = torch.min(all_energies_tensor)  # []

                selected_energies = [energy for energy in all_energies]

                cluster_loss = 0.0
                if len(selected_energies) > 0:
                    selected_energies_tensor = torch.stack(selected_energies, dim=0)
                    cluster_loss = torch.sum(torch.abs(selected_energies_tensor - min_energy)) / len(selected_energies)
                    total_loss += cluster_loss

        if self.num_clusters > 0:
            return total_loss / self.num_clusters
        else:
            return torch.tensor(0.0, device=device)