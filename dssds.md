pip install deepspeed
pip install sentencepiece
pip install xformers
pip install mpi4py



以baichuan项目为基础，进行书写代码，目的只有一个，能够让baichuan项目从零进行训练GPT
2、增加 从零进行训练模型、从训练好的检查点继续进行训练、通过args进行设置
2.1如果 init_from 为 'scratch'，则表示从头开始训练一个新模型。代码会使用给定的参数初始化一个新的 GPT 模型。
2.2如果 init_from 为 'resume'，则表示从之前的检查点中恢复训练。代码会加载之前保存的检查点，并使用检查点中保存的模型参数继续训练。
3、增加AMP功能，通过通过args进行设置数据类型，float16 和 bfloat16  都会启用了混合精度训练的 GradScaler 对象。
3.1 在nanogpt的train.py中，设置数据类型，float16 和 bfloat16，则为了梯度爆炸的防范主要依赖于梯度裁剪，也加入在代码中
4、在训练中，通过设置args设置多少epoch进行设置每多少epoch 进行保存检查点。默认为10，每次10的倍数则保存检查点
5、次10的倍数则保存检查点时，生成epoch01、epoch02进行保存检查点 ，并且保存 config.json
6、如果 init_from 为 'resume'，则表示从之前的检查点中恢复训练。则从检查点继续恢复训练，注意检查点是在epoch02文件中进行存储的
7、模型剪枝和压缩：nanogpt可能会提供模型剪枝和压缩的功能，以减少模型的大小和计算量。Baichuan项目可以考虑添加类似的功能，以提高模型的部署和使用效率。
8、compile：这个参数用于指定是否使用PyTorch 2.0来编译模型以提高性能， 通过args指定
10、然后梳理整个项目，将查看代码结构是否存在不合理或者缺少一些基础功能，进行丰富。
11、代码中的所有输出提示都使用中文，


参考nanogptde 的项目，进行书写代码train.py,请注意，在从零进行训练时，必须使用baichaun自己的模型modeling_baichuan.py。



