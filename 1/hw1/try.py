import hw1 as hw
import numpy as np

td,tl,vd,vl,tsd,tsl = hw.load_mnist_data_file("./../data/")
init=0
no=5
weight_init=hw.random_normal_weight_init
bias_init=hw.zeros_bias_init

mlp = hw.MLP(784, 10, [], [hw.Identity()], weight_init, bias_init,
                      hw.SoftmaxCrossEntropy(), 0.008, momentum=0.0,
                      num_bn_layers=0)
z = mlp.forward(td[init:init+no])

labels = np.array([np.zeros(10) for l in tl])
for i in range(tl.shape[0]):
	labels[i][tl[i]]=1
mlp.backward(labels[init:init+no])
print(mlp.dW[0].shape)
#print("Actual:{} Prediction:{} Loss:{}".format(tl[init:init+no],z.argmax(1),loss))
#print("Predict_Accuracy:{} Loss_zero:{}".format(tl[init:init+no]==z.argmax(1),loss==0))
#print("Loss_accuracy:{}".format((tl[init:init+no]==z.argmax(1))==(loss==0)))



mlp1 = hw.MLP(784, 10, [32], [hw.Sigmoid(), hw.Identity()], weight_init, bias_init, hw.SoftmaxCrossEntropy(), 0.008, momentum=0.0, num_bn_layers=0)
mlp1.forward(td[init:init+no])
mlp1.backward(labels[init:init+no])


ref=hw
hw1=hw
mlp2 = hw.MLP(784, 10, [64, 32], [ref.Sigmoid(), ref.Sigmoid(), hw1.Identity()],
                      weight_init, bias_init, ref.SoftmaxCrossEntropy(), 0.008,
                      momentum=0.0, num_bn_layers=0)

mlp3 = hw1.MLP(784, 10, [32, 32, 32, 32, 32],
                      [ref.Sigmoid(), ref.Sigmoid(), ref.Sigmoid(), ref.Sigmoid(),
                       ref.Sigmoid(), hw1.Identity()],
                      weight_init, bias_init, ref.SoftmaxCrossEntropy(), 0.008,
                      momentum=0.0, num_bn_layers=0)