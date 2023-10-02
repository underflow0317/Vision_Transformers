# Vision_Transformers
基於Google Research, Brain Team  
  
在version1中，只使用cifar-10做train and test，test accuracy約55%，如 Dosovitskiy et al.,2021 所述，在小規模訓練集下，performance不及cnn。  
在version2中，用coco做pre-train，用cifar-10做fine-tune，在rtx 2060上會OOM，還須除錯。

Ref.  
Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby. AN IMAGE IS WORTH 16X16 WORDS:TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE . In ICLR, 2021.
