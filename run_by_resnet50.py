import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from load_data import load_train, BuildDataset, TestDataset
from torch.utils.tensorboard import SummaryWriter
import time
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.models as models

#

data_root = './dataset/'
train_data = data_root + 'train_images'
img_id, labels = load_train(data_root)
dataset = BuildDataset(file_paths=train_data, labels=labels, img_id=img_id)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

print(f'train_samples：{train_size}')
print(f'test_samples：{val_size}')

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

test = TestDataset(data_root + 'test_images')

batch_size = 8
batch_eval = 16

dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dataloader_eval = torch.utils.data.DataLoader(val_dataset, batch_size=batch_eval, shuffle=False)

model = models.resnet50(weights=None)
model.to('cuda')

transform = nn.Sequential(
    nn.Linear(1000, 64),
    nn.Linear(64, 10),
    nn.Linear(10, 5)).cuda()

loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()
lr = 0.00001
optimizer = torch.optim.Adam(list(model.parameters())+list(transform.parameters()), lr=lr)

epoch = 50
bar_length = 100
T_max = 25
eta_min = 0
scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

writer = SummaryWriter('./log_train')

total_test_step = 0
for i in range(epoch):
    print('----------epoch_{} start----------'.format(i+1))
    start_time = time.time()
    total_train_step = 0
    for data in dataloader_train:

        imgs, labels = data
        targets = labels
        outputs = model(imgs)
        results = transform(outputs)
        loss = loss_fn(results, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        scheduler.step()
        progress = total_train_step * batch_size / train_size
        hashes = '#' * int(progress * bar_length)
        current_time = time.time()
        if total_train_step > 0:
            average_time_per_step = (current_time - start_time) / total_train_step / batch_size
            remaining_samples = train_size - (total_train_step * batch_size)
            remaining_time = remaining_samples * average_time_per_step
        else:
            remaining_time = 0
        remaining_time_hours = int(remaining_time // 3600)
        remaining_time_minutes = int((remaining_time % 3600) // 60)
        remaining_time_seconds = int(remaining_time % 60)

        remaining_time_formatted = f"{remaining_time_hours:02}:{remaining_time_minutes:02}:{remaining_time_seconds:02}" if total_train_step < 2140 else remaining_time_formatted = '00:00:00'
        print(f'\repoch_{i+1}: [{hashes:<{bar_length}}] {int(progress * 100)}% ({total_train_step}/{int(train_size/batch_size)+1})\
         eta：{remaining_time_formatted}', end='')
        if total_train_step % 100 == 0:
            print(f"\nprocess：{total_train_step}/{int(train_size/batch_size)+1}，loss:{loss.item()}")
            writer.add_scalar(f'train_loss in epoch{i+1}', loss.item(), total_train_step)
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        print(f'\n----------eval_{i+1} start----------')
        for data in dataloader_eval:
            imgs, labels = data
            targets = labels
            outputs = model(imgs)
            results = transform(outputs)
            loss = loss_fn(results, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (results.argmax(1) == targets).sum()
            total_accuracy += accuracy
    total_test_step += 1
    print(f'\nloss on test data: {total_test_loss * batch_eval / val_size}')
    print(f'\naccuracy on test data: {total_accuracy / val_size}')
    writer.add_scalar('test_accuracy', total_accuracy/val_size, total_test_step)
    writer.add_scalar('test_loss', total_test_loss * batch_eval / val_size, total_test_step)

    torch.save(model, f'./model/epoch_{i+1}.pth')
writer.close()
