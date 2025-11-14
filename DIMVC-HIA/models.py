from loss import *
from metrics import *
from dataprocessing import *


def pre_train(network_model, mv_data, batch_size, epochs, optimizer):
    t = time.time()
    mv_data_loader, num_views, num_samples, _ = get_multiview_data(mv_data, batch_size)

    pre_train_loss_values = np.zeros(epochs, dtype=np.float64)

    criterion = torch.nn.MSELoss()
    for epoch in range(epochs):
        total_loss = 0.
        for batch_idx, (sub_data_views, _) in enumerate(mv_data_loader):
            _, dvs, _, _, _,_ = network_model(sub_data_views)
            loss_list = list()
            for idx in range(num_views):
                mask = ((sub_data_views[idx] != 0)).any(dim=1).unsqueeze(1).float()
                loss_list.append(criterion(sub_data_views[idx] * mask, dvs[idx] * mask))
            loss = sum(loss_list)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        pre_train_loss_values[epoch] = total_loss
        if epoch % 10 == 0 or epoch == epochs - 1:
            print('Pre-training, epoch {}, Loss:{:.7f}'.format(epoch, total_loss / num_samples))

    print("Pre-training finished.")
    print("Total time elapsed: {:.4f}s".format(time.time() - t))

    return pre_train_loss_values


def contrastive_train(network_model, mv_data, mvc_loss, batch_size, beta, alpha, gamma, temperature_l, normalized, epoch,
                      optimizer):

    network_model.train()
    mv_data_loader, num_views, num_samples, num_clusters = get_multiview_data(mv_data, batch_size)
    criterion = torch.nn.MSELoss()
    total_loss = 0.
    for batch_idx, (sub_data_views, _) in enumerate(mv_data_loader):
        lbps, dvs, _, ebm_loss, _, view_sims = network_model(sub_data_views)
        loss_list = list()
        view_sim_idx = 0
        for i in range(num_views):
            for j in range(i + 1, num_views):
                current_view_sim = view_sims[view_sim_idx]
                loss_list.append(current_view_sim * beta * mvc_loss.forward_label(lbps[i], lbps[j], temperature_l, normalized))
                loss_list.append(current_view_sim * beta * mvc_loss.forward_prob(lbps[i], lbps[j]))
                view_sim_idx += 1
            loss_list.append(gamma * criterion(sub_data_views[i], dvs[i]))
        loss = sum(loss_list)
        loss += alpha * ebm_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 10 == 0:
        print('Contrastive_train, epoch {} loss:{:.7f}'.format(epoch, total_loss / num_samples))

    return total_loss


def inference(network_model, mv_data, batch_size):

    network_model.eval()
    mv_data_loader, num_views, num_samples, _ = get_multiview_data(mv_data, batch_size)

    soft_vector = []
    pred_vectors = []
    labels_vector = []
    TSNE_features_list = []
    for v in range(num_views):
        pred_vectors.append([])

    for batch_idx, (sub_data_views, sub_labels) in enumerate(mv_data_loader):
        with torch.no_grad():
            lbps, _, _, _, _,_ = network_model(sub_data_views)
            lbp = sum(lbps)/num_views

        for idx in range(num_views):
            pred_label = torch.argmax(lbps[idx], dim=1)
            pred_vectors[idx].extend(pred_label.detach().cpu().numpy())

        soft_vector.extend(lbp.detach().cpu().numpy())
        labels_vector.extend(sub_labels)

    for idx in range(num_views):
        pred_vectors[idx] = np.array(pred_vectors[idx])

    actual_num_samples = len(soft_vector)
    labels_vector = np.array(labels_vector).reshape(actual_num_samples)
    total_pred = np.argmax(np.array(soft_vector), axis=1)

    return total_pred, pred_vectors, labels_vector

def valid(network_model, mv_data, batch_size):

    total_pred, pred_vectors, labels_vector = inference(network_model, mv_data, batch_size)
    num_views = len(mv_data.data_views)

    print("Clustering results on cluster assignments of each view:")
    for idx in range(num_views):
        acc, nmi, pur, ari = calculate_metrics(labels_vector,  pred_vectors[idx])
        print('ACC{} = {:.4f} NMI{} = {:.4f} PUR{} = {:.4f} ARI{}={:.4f}'.format(idx+1, acc,
                                                                                 idx+1, nmi,
                                                                                 idx+1, pur,
                                                                                 idx+1, ari))

    print("Clustering results on semantic labels: " + str(labels_vector.shape[0]))
    acc, nmi, pur, ari = calculate_metrics(labels_vector, total_pred)
    print('ACC = {:.4f} NMI = {:.4f} PUR = {:.4f} ARI={:.4f}'.format(acc, nmi, pur, ari))

    return acc, nmi, pur, ari