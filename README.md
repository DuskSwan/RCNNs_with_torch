个人学习记录
=======
这是一个个人学习《PyTorch教程：21个项目玩转PyTorch实战》（第五章）的记录

书中的系统是Linux，而我想要在win10系统的个人电脑上实现运行。而且我的个人电脑没有nvidia显卡，只能用CPU跑。

### 相比原书的改动

1. 我希望项目更加便于维护，结构更加清楚，所以修改了目录结构。随书附带的资源只有一个脚本，下载下来的工具文件与该脚本直接共存于工作目录下，我将GitHub处下载的文件放在了目录packs下，以包的形式调用。
1. 函数get_model_instance_segmentation中会使用torchvision提供的预训练模型，Windows下的默认下载路径是`C:\Users\<username>\.cache`，于是我为这个函数加了一个参数来指定下载的路径。
   （参考https://blog.csdn.net/yanxiangtianji/article/details/112256618）
1. 因为没有GPU，所以engine.py中，函数evaluate中的`torch.cuda.synchronize()`一行要改为`torch.cuda.synchronize() if device.type == 'cuda' else None`，否则报错。
