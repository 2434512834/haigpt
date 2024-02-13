
# tiny shakespeare

Tiny shakespeare, of the good old char-rnn fame :)

After running `prepare.py`:

- train.bin has 301,966 tokens
- val.bin has 36,059 tokens

为什么nanogpt的这个代码中，没有词汇表呢



这个代码是用于准备字符级语言模型的数据，而不是词级语言模型。在字符级语言模型中，模型预测的是下一个字符，而不是下一个单词。因此，它的“词汇表”实际上就是所有可能的字符集合。

在这个特定的代码中，`prepare.py` 脚本将莎士比亚的文本转换为一系列的字符 ID。这些 ID 是通过将每个唯一字符映射到一个唯一的整数来得到的。这个映射就是字符级语言模型的“词汇表”。

所以，虽然这个代码中没有明显的“词汇表”，但实际上，它是通过字符到 ID 的映射来隐式地创建了一个词汇表。希望这个答案能帮到你！如果你还有其他问题，欢迎随时向我提问。😊


