import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms

#
def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 50000张训练图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=False, transform=transform)#转换成torch
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
                                               shuffle=True, num_workers=0)#num_workers=几个核心

    # 10000张验证图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=5000,
                                             shuffle=False, num_workers=0)
    val_data_iter = iter(val_loader)
    val_image, val_label = val_data_iter.next()

    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = LeNet().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)#优化器 学习

    for epoch in range(50):  # loop over the dataset multiple times

        running_loss = 0.0#训练中的损失
        for step, data in enumerate(train_loader, start=0):#训练集赝本和步数
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs,labels = inputs.to(device),labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()#可以累计banch 32*3=96
            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_function(outputs, labels)#网络预测数值 标签
            loss.backward()#反向传播
            optimizer.step()#更新

            # print statistics
            running_loss += loss.item()
            if step % 500 == 499:    # print every 500 mini-batches
                with torch.no_grad():#with 上下文管理器 torch.no_grad()避免卡顿 训练中就已经计算了误差梯度
                    outputs = net(val_image)  # [batch, 10]
                    predict_y = torch.max(outputs, dim=1)[1]#找到位置 10个找到最大值
                    accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)#item拿到数值

                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))
                    running_loss = 0.0

    print('Finished Training')

    save_path = './Lenet.pth'
    torch.save(net.state_dict(), save_path)#保存参数和路径
#
#
if __name__ == '__main__':
    main()
