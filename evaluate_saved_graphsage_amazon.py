import torch
from graphsage_amazon import GraphSAGEAmazon
from data_loader_amazon import load_amazon_data
from cold_validation_amazon import evaluate_graphsage_amazon

CHECKPOINT_PATH = 'best_graphsage_amazon.pt'


def main():
    data = load_amazon_data()
    model = GraphSAGEAmazon(
        num_users=data['num_users'],
        item_input_dim=384,
        hidden_dim=64,
    )
    state = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    print('Loaded baseline checkpoint:', CHECKPOINT_PATH, flush=True)
    print('Evaluating cold validation only...', flush=True)
    metrics = evaluate_graphsage_amazon(model)
    for name in ('Recall@5', 'NDCG@5', 'Recall@10', 'NDCG@10', 'Recall@20', 'NDCG@20'):
        print(f'{name}: {float(metrics[name]):.9f}')


if __name__ == '__main__':
    main()
