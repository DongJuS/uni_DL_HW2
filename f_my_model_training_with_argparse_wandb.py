import torch
from torch import nn, optim
from torch.utils.data import Dataset, random_split, DataLoader
from datetime import datetime
import wandb
import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from pathlib import Path
import sys

# BASE_PATH 설정 (필요에 따라 수정)
BASE_PATH = str(Path(__file__).resolve().parent.parent.parent)  # 예: /Users/yhhan/git/link_dl
print(BASE_PATH, "!!!!!")
sys.path.append(BASE_PATH)

# 데이터셋 클래스 정의
class TitanicDataset(Dataset):
    def __init__(self, csv_file, is_test=False):
        self.data = pd.read_csv(csv_file)

        if is_test:
            self.passenger_ids = self.data['PassengerId']
            # 테스트 데이터셋에서는 'Survived' 열이 없으므로 제거하지 않습니다.
            columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked']
            self.data = self.data.drop(columns=columns_to_drop, axis=1, errors='ignore')
        else:
            # 훈련 데이터셋에서는 'Survived' 열이 필요하므로 제거하지 않습니다.
            columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked']
            self.data = self.data.drop(columns=columns_to_drop, axis=1, errors='ignore')

        # 'Sex' 인코딩: male=0, female=1
        self.data['Sex'] = self.data['Sex'].map({'male': 0, 'female': 1})

        # 'Age' 결측값을 평균으로 채움
        self.data['Age'] = self.data['Age'].fillna(self.data['Age'].mean())

        if not is_test:
            # 입력 특징과 타겟 분리
            self.X = self.data[['Pclass', 'Sex', 'Age']].values.astype(float)
            self.y = self.data['Survived'].values.astype(float)
        else:
            # 테스트 데이터셋에서는 타겟이 없음
            self.X = self.data[['Pclass', 'Sex', 'Age']].values.astype(float)
            self.y = None

        # 정규화
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)

        # 텐서로 변환
        self.X = torch.tensor(self.X, dtype=torch.float32)
        if not is_test:
            self.y = torch.tensor(self.y, dtype=torch.float32).unsqueeze(1)  # (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            # 테스트 데이터셋에서는 PassengerId와 함께 반환
            return self.X[idx], self.passenger_ids.iloc[idx]

def get_data(csv_path):
    dataset = TitanicDataset(csv_file=csv_path, is_test=False)
    print(dataset)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, val_size])
    print(f"Train size: {len(train_dataset)}, Validation size: {len(validation_dataset)}")

    train_data_loader = DataLoader(dataset=train_dataset, batch_size=wandb.config.batch_size, shuffle=True)
    validation_data_loader = DataLoader(dataset=validation_dataset, batch_size=len(validation_dataset))

    return train_data_loader, validation_data_loader

def get_test_data(csv_path):
    test_dataset = TitanicDataset(csv_file=csv_path, is_test=True)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset))
    return test_data_loader

class MyModel(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(n_input, wandb.config.n_hidden_unit_list[0]),
            nn.ELU(),  # ReLU에서 ELU로 변경
            nn.Linear(wandb.config.n_hidden_unit_list[0], wandb.config.n_hidden_unit_list[1]),
            nn.ELU(),  # ReLU에서 ELU로 변경
            nn.Linear(wandb.config.n_hidden_unit_list[1], n_output),
            nn.Sigmoid()  # 이진 분류를 위해 Sigmoid 활성화 추가
        )

    def forward(self, x):
        return self.model(x)

def get_model_and_optimizer():
    my_model = MyModel(n_input=3, n_output=1)  # 입력 특징 수에 맞게 조정
    optimizer = optim.Adam(my_model.parameters(), lr=wandb.config.learning_rate)  # Adam 옵티마이저 사용

    return my_model, optimizer

def training_loop(model, optimizer, train_data_loader, validation_data_loader):
    n_epochs = wandb.config.epochs
    loss_fn = nn.BCELoss()  # 이진 분류를 위한 손실 함수
    next_print_epoch = 100

    for epoch in range(1, n_epochs + 1):
        model.train()
        loss_train = 0.0
        all_preds_train = []
        all_targets_train = []

        for train_batch in train_data_loader:
            input, target = train_batch
            output_train = model(input)
            loss = loss_fn(output_train, target)
            loss_train += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = (output_train > 0.5).float()
            all_preds_train.extend(preds.cpu().numpy())
            all_targets_train.extend(target.cpu().numpy())

        train_loss = loss_train / len(train_data_loader)
        train_accuracy = accuracy_score(all_targets_train, all_preds_train)
        train_precision = precision_score(all_targets_train, all_preds_train)
        train_recall = recall_score(all_targets_train, all_preds_train)
        train_f1 = f1_score(all_targets_train, all_preds_train)

        model.eval()
        loss_validation = 0.0
        all_preds_val = []
        all_targets_val = []

        with torch.no_grad():
            for validation_batch in validation_data_loader:
                input, target = validation_batch
                output_validation = model(input)
                loss = loss_fn(output_validation, target)
                loss_validation += loss.item()

                preds = (output_validation > 0.5).float()
                all_preds_val.extend(preds.cpu().numpy())
                all_targets_val.extend(target.cpu().numpy())

        val_loss = loss_validation / len(validation_data_loader)
        val_accuracy = accuracy_score(all_targets_val, all_preds_val)
        val_precision = precision_score(all_targets_val, all_preds_val)
        val_recall = recall_score(all_targets_val, all_preds_val)
        val_f1 = f1_score(all_targets_val, all_preds_val)

        wandb.log({
            "Epoch": epoch,
            "Training Loss": train_loss,
            "Validation Loss": val_loss,
            "Training Accuracy": train_accuracy,
            "Validation Accuracy": val_accuracy,
            "Training Precision": train_precision,
            "Validation Precision": val_precision,
            "Training Recall": train_recall,
            "Validation Recall": val_recall,
            "Training F1 Score": train_f1,
            "Validation F1 Score": val_f1
        })

        if epoch >= next_print_epoch:
            print(
                f"Epoch {epoch}, "
                f"Training Loss: {train_loss:.4f}, "
                f"Validation Loss: {val_loss:.4f}, "
                f"Training Acc: {train_accuracy:.4f}, "
                f"Validation Acc: {val_accuracy:.4f}, "
                f"Training Precision: {train_precision:.4f}, "
                f"Validation Precision: {val_precision:.4f}, "
                f"Training Recall: {train_recall:.4f}, "
                f"Validation Recall: {val_recall:.4f}, "
                f"Training F1: {train_f1:.4f}, "
                f"Validation F1: {val_f1:.4f}"
            )
            next_print_epoch += 100

# def predict_test(model, test_data_loader, output_path):
#     model.eval()
#     all_preds = []
#     all_passenger_ids = []
#
#     with torch.no_grad():
#         for test_batch in test_data_loader:
#             input, passenger_ids = test_batch
#             output = model(input)
#             preds = (output > 0.5).int()
#             # preds를 1차원으로 변환하여 리스트에 추가
#             all_preds.extend(preds.cpu().numpy().flatten())
#             all_passenger_ids.extend(passenger_ids.cpu().numpy())
#
#     # 결과를 CSV 파일로 저장
#     preds_df = pd.DataFrame({
#         'PassengerId': all_passenger_ids,
#         'Survived': all_preds
#     })
#     preds_df.to_csv(output_path, index=False)
#     print(f"Test predictions saved to {output_path}")
def predict_test(model, test_data_loader, output_path):
    model.eval()
    all_preds = []
    all_passenger_ids = []

    with torch.no_grad():
        for test_batch in test_data_loader:
            input, passenger_ids = test_batch
            output = model(input)
            preds = (output > 0.5).int()
            # preds를 1차원으로 변환하여 리스트에 추가
            all_preds.extend(preds.cpu().numpy().flatten())
            all_passenger_ids.extend(passenger_ids.cpu().numpy())

    # 결과를 CSV 파일로 저장
    preds_df = pd.DataFrame({
        'PassengerId': all_passenger_ids,
        'Survived': all_preds
    })
    preds_df.to_csv(output_path, index=False)
    print(f"Test predictions saved to {output_path}")


def main(args):
    current_time_str = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')

    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'n_hidden_unit_list': args.hidden_units,  # 리스트로 받도록 수정
    }

    wandb.init(
        mode="online" if args.wandb else "disabled",
        project="titanic_model_training",
        notes="Titanic survival prediction experiment",
        tags=["titanic", "binary_classification"],
        name=current_time_str,
        config=config
    )
    print(args)
    print(wandb.config)

    # 훈련 및 검증 데이터 로더
    train_data_loader, validation_data_loader = get_data(csv_path=args.train_csv_path)

    # 모델 및 옵티마이저 초기화
    model, optimizer = get_model_and_optimizer()

    # 훈련 루프 실행
    training_loop(
        model=model,
        optimizer=optimizer,
        train_data_loader=train_data_loader,
        validation_data_loader=validation_data_loader
    )

    # 테스트 데이터 로더 (타겟 없음)
    test_data_loader = get_test_data(csv_path=args.test_csv_path)

    # 테스트 데이터에 대한 예측 수행 및 저장
    predict_test(model, test_data_loader, output_path=args.output_path)

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--wandb", action=argparse.BooleanOptionalAction, default=False, help="Enable or disable WandB logging"
    )

    parser.add_argument(
        "-b", "--batch_size", type=int, default=32, help="Batch size (int, default: 32)"
    )

    parser.add_argument(
        "-e", "--epochs", type=int, default=100, help="Number of training epochs (int, default: 100)"
    )

    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate (float, default: 1e-3)"
    )

    parser.add_argument(
        "--hidden_units", type=int, nargs='+', default=[64, 32], help="List of hidden units (default: [64, 32])"
    )

    # 수정된 인자
    parser.add_argument(
        "--train_csv_path", type=str, required=True, help="Path to the training CSV dataset"
    )

    parser.add_argument(
        "--test_csv_path", type=str, required=True, help="Path to the testing CSV dataset"
    )

    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save test predictions"
    )

    args = parser.parse_args()

    main(args)
