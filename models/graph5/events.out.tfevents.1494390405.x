       £K"	  @°•D÷Abrain.Event:2ЂV¶Ѕ†     ґыєэ	е+b°•D÷A"У±
a
inputs/graphsPlaceholder*
shape: *
dtype0*)
_output_shapes
:€€€€€€€€€АА
]
inputs/typePlaceholder*'
_output_shapes
:€€€€€€€€€*
shape: *
dtype0
v
prepare_tensors/Reshape/shapeConst*%
valueB"€€€€А   А      *
_output_shapes
:*
dtype0
Ъ
prepare_tensors/ReshapeReshapeinputs/graphsprepare_tensors/Reshape/shape*
Tshape0*1
_output_shapes
:€€€€€€€€€АА*
T0
~
%conv_1/weights/truncated_normal/shapeConst*%
valueB"         
   *
dtype0*
_output_shapes
:
i
$conv_1/weights/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
k
&conv_1/weights/truncated_normal/stddevConst*
valueB
 *Ќћћ=*
dtype0*
_output_shapes
: 
ј
/conv_1/weights/truncated_normal/TruncatedNormalTruncatedNormal%conv_1/weights/truncated_normal/shape*

seed *
T0*
dtype0*&
_output_shapes
:
*
seed2 
і
#conv_1/weights/truncated_normal/mulMul/conv_1/weights/truncated_normal/TruncatedNormal&conv_1/weights/truncated_normal/stddev*&
_output_shapes
:
*
T0
Ґ
conv_1/weights/truncated_normalAdd#conv_1/weights/truncated_normal/mul$conv_1/weights/truncated_normal/mean*&
_output_shapes
:
*
T0
Ы
conv_1/weights/W_conv_1
VariableV2*
shape:
*
shared_name *
dtype0*&
_output_shapes
:
*
	container 
и
conv_1/weights/W_conv_1/AssignAssignconv_1/weights/W_conv_1conv_1/weights/truncated_normal**
_class 
loc:@conv_1/weights/W_conv_1*&
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
Ю
conv_1/weights/W_conv_1/readIdentityconv_1/weights/W_conv_1**
_class 
loc:@conv_1/weights/W_conv_1*&
_output_shapes
:
*
T0
`
conv_1/biases/ConstConst*
valueB
*Ќћћ=*
_output_shapes
:
*
dtype0
В
conv_1/biases/b_conv_1
VariableV2*
shape:
*
shared_name *
dtype0*
_output_shapes
:
*
	container 
Ќ
conv_1/biases/b_conv_1/AssignAssignconv_1/biases/b_conv_1conv_1/biases/Const*
use_locking(*
T0*)
_class
loc:@conv_1/biases/b_conv_1*
validate_shape(*
_output_shapes
:

П
conv_1/biases/b_conv_1/readIdentityconv_1/biases/b_conv_1*)
_class
loc:@conv_1/biases/b_conv_1*
_output_shapes
:
*
T0
ь
(conv_1/convolution_and_non_linear/Conv2DConv2Dprepare_tensors/Reshapeconv_1/weights/W_conv_1/read*
paddingSAME*
T0*
strides
*
data_formatNHWC*1
_output_shapes
:€€€€€€€€€АА
*
use_cudnn_on_gpu(
ѓ
%conv_1/convolution_and_non_linear/addAdd(conv_1/convolution_and_non_linear/Conv2Dconv_1/biases/b_conv_1/read*
T0*1
_output_shapes
:€€€€€€€€€АА

С
&conv_1/convolution_and_non_linear/ReluRelu%conv_1/convolution_and_non_linear/add*
T0*1
_output_shapes
:€€€€€€€€€АА

“
conv_1/pool/MaxPoolMaxPool&conv_1/convolution_and_non_linear/Relu*
ksize
*/
_output_shapes
:€€€€€€€€€  
*
T0*
strides
*
data_formatNHWC*
paddingSAME
~
%conv_2/weights/truncated_normal/shapeConst*%
valueB"      
      *
dtype0*
_output_shapes
:
i
$conv_2/weights/truncated_normal/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
k
&conv_2/weights/truncated_normal/stddevConst*
valueB
 *Ќћћ=*
_output_shapes
: *
dtype0
ј
/conv_2/weights/truncated_normal/TruncatedNormalTruncatedNormal%conv_2/weights/truncated_normal/shape*&
_output_shapes
:
*
seed2 *
T0*

seed *
dtype0
і
#conv_2/weights/truncated_normal/mulMul/conv_2/weights/truncated_normal/TruncatedNormal&conv_2/weights/truncated_normal/stddev*&
_output_shapes
:
*
T0
Ґ
conv_2/weights/truncated_normalAdd#conv_2/weights/truncated_normal/mul$conv_2/weights/truncated_normal/mean*
T0*&
_output_shapes
:

Ы
conv_2/weights/W_conv_2
VariableV2*&
_output_shapes
:
*
	container *
shape:
*
dtype0*
shared_name 
и
conv_2/weights/W_conv_2/AssignAssignconv_2/weights/W_conv_2conv_2/weights/truncated_normal**
_class 
loc:@conv_2/weights/W_conv_2*&
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
Ю
conv_2/weights/W_conv_2/readIdentityconv_2/weights/W_conv_2*
T0**
_class 
loc:@conv_2/weights/W_conv_2*&
_output_shapes
:

`
conv_2/biases/ConstConst*
valueB*Ќћћ=*
dtype0*
_output_shapes
:
В
conv_2/biases/b_conv_2
VariableV2*
_output_shapes
:*
	container *
shape:*
dtype0*
shared_name 
Ќ
conv_2/biases/b_conv_2/AssignAssignconv_2/biases/b_conv_2conv_2/biases/Const*)
_class
loc:@conv_2/biases/b_conv_2*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
П
conv_2/biases/b_conv_2/readIdentityconv_2/biases/b_conv_2*)
_class
loc:@conv_2/biases/b_conv_2*
_output_shapes
:*
T0
ц
(conv_2/convolution_and_non_linear/Conv2DConv2Dconv_1/pool/MaxPoolconv_2/weights/W_conv_2/read*/
_output_shapes
:€€€€€€€€€  *
T0*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC*
paddingSAME
≠
%conv_2/convolution_and_non_linear/addAdd(conv_2/convolution_and_non_linear/Conv2Dconv_2/biases/b_conv_2/read*/
_output_shapes
:€€€€€€€€€  *
T0
П
&conv_2/convolution_and_non_linear/ReluRelu%conv_2/convolution_and_non_linear/add*
T0*/
_output_shapes
:€€€€€€€€€  
“
conv_2/pool/MaxPoolMaxPool&conv_2/convolution_and_non_linear/Relu*
paddingSAME*
T0*
strides
*
data_formatNHWC*/
_output_shapes
:€€€€€€€€€*
ksize

~
%conv_3/weights/truncated_normal/shapeConst*%
valueB"         (   *
_output_shapes
:*
dtype0
i
$conv_3/weights/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
k
&conv_3/weights/truncated_normal/stddevConst*
valueB
 *Ќћћ=*
dtype0*
_output_shapes
: 
ј
/conv_3/weights/truncated_normal/TruncatedNormalTruncatedNormal%conv_3/weights/truncated_normal/shape*&
_output_shapes
:(*
seed2 *
T0*

seed *
dtype0
і
#conv_3/weights/truncated_normal/mulMul/conv_3/weights/truncated_normal/TruncatedNormal&conv_3/weights/truncated_normal/stddev*
T0*&
_output_shapes
:(
Ґ
conv_3/weights/truncated_normalAdd#conv_3/weights/truncated_normal/mul$conv_3/weights/truncated_normal/mean*&
_output_shapes
:(*
T0
Ы
conv_3/weights/W_conv_3
VariableV2*
shape:(*
shared_name *
dtype0*&
_output_shapes
:(*
	container 
и
conv_3/weights/W_conv_3/AssignAssignconv_3/weights/W_conv_3conv_3/weights/truncated_normal*
use_locking(*
T0**
_class 
loc:@conv_3/weights/W_conv_3*
validate_shape(*&
_output_shapes
:(
Ю
conv_3/weights/W_conv_3/readIdentityconv_3/weights/W_conv_3**
_class 
loc:@conv_3/weights/W_conv_3*&
_output_shapes
:(*
T0
`
conv_3/biases/ConstConst*
valueB(*Ќћћ=*
dtype0*
_output_shapes
:(
В
conv_3/biases/b_conv_3
VariableV2*
_output_shapes
:(*
	container *
shape:(*
dtype0*
shared_name 
Ќ
conv_3/biases/b_conv_3/AssignAssignconv_3/biases/b_conv_3conv_3/biases/Const*
use_locking(*
T0*)
_class
loc:@conv_3/biases/b_conv_3*
validate_shape(*
_output_shapes
:(
П
conv_3/biases/b_conv_3/readIdentityconv_3/biases/b_conv_3*)
_class
loc:@conv_3/biases/b_conv_3*
_output_shapes
:(*
T0
ц
(conv_3/convolution_and_non_linear/Conv2DConv2Dconv_2/pool/MaxPoolconv_3/weights/W_conv_3/read*
paddingSAME*
T0*
strides
*
data_formatNHWC*/
_output_shapes
:€€€€€€€€€(*
use_cudnn_on_gpu(
≠
%conv_3/convolution_and_non_linear/addAdd(conv_3/convolution_and_non_linear/Conv2Dconv_3/biases/b_conv_3/read*/
_output_shapes
:€€€€€€€€€(*
T0
П
&conv_3/convolution_and_non_linear/ReluRelu%conv_3/convolution_and_non_linear/add*
T0*/
_output_shapes
:€€€€€€€€€(
“
conv_3/pool/MaxPoolMaxPool&conv_3/convolution_and_non_linear/Relu*
ksize
*/
_output_shapes
:€€€€€€€€€(*
T0*
strides
*
data_formatNHWC*
paddingSAME
f
flatten/Reshape/shapeConst*
valueB"€€€€ 
  *
_output_shapes
:*
dtype0
З
flatten/ReshapeReshapeconv_3/pool/MaxPoolflatten/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
t
#fc_1/weights/truncated_normal/shapeConst*
valueB" 
  d   *
_output_shapes
:*
dtype0
g
"fc_1/weights/truncated_normal/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
i
$fc_1/weights/truncated_normal/stddevConst*
valueB
 *Ќћћ=*
_output_shapes
: *
dtype0
µ
-fc_1/weights/truncated_normal/TruncatedNormalTruncatedNormal#fc_1/weights/truncated_normal/shape*
_output_shapes
:	Аd*
seed2 *
T0*

seed *
dtype0
І
!fc_1/weights/truncated_normal/mulMul-fc_1/weights/truncated_normal/TruncatedNormal$fc_1/weights/truncated_normal/stddev*
_output_shapes
:	Аd*
T0
Х
fc_1/weights/truncated_normalAdd!fc_1/weights/truncated_normal/mul"fc_1/weights/truncated_normal/mean*
T0*
_output_shapes
:	Аd
И
fc_1/weights/W_fc1
VariableV2*
shape:	Аd*
shared_name *
dtype0*
_output_shapes
:	Аd*
	container 
–
fc_1/weights/W_fc1/AssignAssignfc_1/weights/W_fc1fc_1/weights/truncated_normal*
use_locking(*
T0*%
_class
loc:@fc_1/weights/W_fc1*
validate_shape(*
_output_shapes
:	Аd
И
fc_1/weights/W_fc1/readIdentityfc_1/weights/W_fc1*%
_class
loc:@fc_1/weights/W_fc1*
_output_shapes
:	Аd*
T0
^
fc_1/biases/ConstConst*
valueBd*Ќћћ=*
_output_shapes
:d*
dtype0
}
fc_1/biases/b_fc1
VariableV2*
shape:d*
shared_name *
dtype0*
_output_shapes
:d*
	container 
Љ
fc_1/biases/b_fc1/AssignAssignfc_1/biases/b_fc1fc_1/biases/Const*
use_locking(*
T0*$
_class
loc:@fc_1/biases/b_fc1*
validate_shape(*
_output_shapes
:d
А
fc_1/biases/b_fc1/readIdentityfc_1/biases/b_fc1*$
_class
loc:@fc_1/biases/b_fc1*
_output_shapes
:d*
T0
≠
!fc_1/matmul_and_non_linear/MatMulMatMulflatten/Reshapefc_1/weights/W_fc1/read*
transpose_b( *'
_output_shapes
:€€€€€€€€€d*
transpose_a( *
T0
Т
fc_1/matmul_and_non_linear/addAdd!fc_1/matmul_and_non_linear/MatMulfc_1/biases/b_fc1/read*'
_output_shapes
:€€€€€€€€€d*
T0
y
fc_1/matmul_and_non_linear/ReluRelufc_1/matmul_and_non_linear/add*'
_output_shapes
:€€€€€€€€€d*
T0
t
#fc_2/weights/truncated_normal/shapeConst*
valueB"d      *
dtype0*
_output_shapes
:
g
"fc_2/weights/truncated_normal/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
i
$fc_2/weights/truncated_normal/stddevConst*
valueB
 *Ќћћ=*
_output_shapes
: *
dtype0
і
-fc_2/weights/truncated_normal/TruncatedNormalTruncatedNormal#fc_2/weights/truncated_normal/shape*
_output_shapes

:d*
seed2 *
T0*

seed *
dtype0
¶
!fc_2/weights/truncated_normal/mulMul-fc_2/weights/truncated_normal/TruncatedNormal$fc_2/weights/truncated_normal/stddev*
T0*
_output_shapes

:d
Ф
fc_2/weights/truncated_normalAdd!fc_2/weights/truncated_normal/mul"fc_2/weights/truncated_normal/mean*
_output_shapes

:d*
T0
Ж
fc_2/weights/W_fc2
VariableV2*
shape
:d*
shared_name *
dtype0*
_output_shapes

:d*
	container 
ѕ
fc_2/weights/W_fc2/AssignAssignfc_2/weights/W_fc2fc_2/weights/truncated_normal*%
_class
loc:@fc_2/weights/W_fc2*
_output_shapes

:d*
T0*
validate_shape(*
use_locking(
З
fc_2/weights/W_fc2/readIdentityfc_2/weights/W_fc2*%
_class
loc:@fc_2/weights/W_fc2*
_output_shapes

:d*
T0
^
fc_2/biases/ConstConst*
valueB*Ќћћ=*
dtype0*
_output_shapes
:
~
fc_2/biases/b_fc_2
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
њ
fc_2/biases/b_fc_2/AssignAssignfc_2/biases/b_fc_2fc_2/biases/Const*%
_class
loc:@fc_2/biases/b_fc_2*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
Г
fc_2/biases/b_fc_2/readIdentityfc_2/biases/b_fc_2*%
_class
loc:@fc_2/biases/b_fc_2*
_output_shapes
:*
T0
™
outputs/MatMulMatMulfc_1/matmul_and_non_linear/Relufc_2/weights/W_fc2/read*
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( 
m
outputs/addAddoutputs/MatMulfc_2/biases/b_fc_2/read*'
_output_shapes
:€€€€€€€€€*
T0
Y
outputs/SoftmaxSoftmaxoutputs/add*
T0*'
_output_shapes
:€€€€€€€€€
[
cross_entorpy/LogLogoutputs/Softmax*'
_output_shapes
:€€€€€€€€€*
T0
j
cross_entorpy/mulMulinputs/typecross_entorpy/Log*
T0*'
_output_shapes
:€€€€€€€€€
m
#cross_entorpy/Sum/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
Ы
cross_entorpy/SumSumcross_entorpy/mul#cross_entorpy/Sum/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:€€€€€€€€€
Y
cross_entorpy/NegNegcross_entorpy/Sum*
T0*#
_output_shapes
:€€€€€€€€€
]
cross_entorpy/ConstConst*
valueB: *
dtype0*
_output_shapes
:
А
cross_entorpy/MeanMeancross_entorpy/Negcross_entorpy/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
|
 cross_entorpy/cross_entorpy/tagsConst*,
value#B! Bcross_entorpy/cross_entorpy*
_output_shapes
: *
dtype0
Г
cross_entorpy/cross_entorpyScalarSummary cross_entorpy/cross_entorpy/tagscross_entorpy/Mean*
T0*
_output_shapes
: 
X
train/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Z
train/gradients/ConstConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
k
train/gradients/FillFilltrain/gradients/Shapetrain/gradients/Const*
_output_shapes
: *
T0

5train/gradients/cross_entorpy/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
Ї
/train/gradients/cross_entorpy/Mean_grad/ReshapeReshapetrain/gradients/Fill5train/gradients/cross_entorpy/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
~
-train/gradients/cross_entorpy/Mean_grad/ShapeShapecross_entorpy/Neg*
T0*
out_type0*
_output_shapes
:
‘
,train/gradients/cross_entorpy/Mean_grad/TileTile/train/gradients/cross_entorpy/Mean_grad/Reshape-train/gradients/cross_entorpy/Mean_grad/Shape*#
_output_shapes
:€€€€€€€€€*
T0*

Tmultiples0
А
/train/gradients/cross_entorpy/Mean_grad/Shape_1Shapecross_entorpy/Neg*
out_type0*
_output_shapes
:*
T0
r
/train/gradients/cross_entorpy/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
w
-train/gradients/cross_entorpy/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
“
,train/gradients/cross_entorpy/Mean_grad/ProdProd/train/gradients/cross_entorpy/Mean_grad/Shape_1-train/gradients/cross_entorpy/Mean_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
y
/train/gradients/cross_entorpy/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
÷
.train/gradients/cross_entorpy/Mean_grad/Prod_1Prod/train/gradients/cross_entorpy/Mean_grad/Shape_2/train/gradients/cross_entorpy/Mean_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
s
1train/gradients/cross_entorpy/Mean_grad/Maximum/yConst*
value	B :*
_output_shapes
: *
dtype0
Њ
/train/gradients/cross_entorpy/Mean_grad/MaximumMaximum.train/gradients/cross_entorpy/Mean_grad/Prod_11train/gradients/cross_entorpy/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
Љ
0train/gradients/cross_entorpy/Mean_grad/floordivFloorDiv,train/gradients/cross_entorpy/Mean_grad/Prod/train/gradients/cross_entorpy/Mean_grad/Maximum*
T0*
_output_shapes
: 
Ц
,train/gradients/cross_entorpy/Mean_grad/CastCast0train/gradients/cross_entorpy/Mean_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0
ƒ
/train/gradients/cross_entorpy/Mean_grad/truedivRealDiv,train/gradients/cross_entorpy/Mean_grad/Tile,train/gradients/cross_entorpy/Mean_grad/Cast*
T0*#
_output_shapes
:€€€€€€€€€
Р
*train/gradients/cross_entorpy/Neg_grad/NegNeg/train/gradients/cross_entorpy/Mean_grad/truediv*#
_output_shapes
:€€€€€€€€€*
T0
}
,train/gradients/cross_entorpy/Sum_grad/ShapeShapecross_entorpy/mul*
out_type0*
_output_shapes
:*
T0
m
+train/gradients/cross_entorpy/Sum_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
®
*train/gradients/cross_entorpy/Sum_grad/addAdd#cross_entorpy/Sum/reduction_indices+train/gradients/cross_entorpy/Sum_grad/Size*
T0*
_output_shapes
:
і
*train/gradients/cross_entorpy/Sum_grad/modFloorMod*train/gradients/cross_entorpy/Sum_grad/add+train/gradients/cross_entorpy/Sum_grad/Size*
_output_shapes
:*
T0
x
.train/gradients/cross_entorpy/Sum_grad/Shape_1Const*
valueB:*
_output_shapes
:*
dtype0
t
2train/gradients/cross_entorpy/Sum_grad/range/startConst*
value	B : *
_output_shapes
: *
dtype0
t
2train/gradients/cross_entorpy/Sum_grad/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
т
,train/gradients/cross_entorpy/Sum_grad/rangeRange2train/gradients/cross_entorpy/Sum_grad/range/start+train/gradients/cross_entorpy/Sum_grad/Size2train/gradients/cross_entorpy/Sum_grad/range/delta*

Tidx0*
_output_shapes
:
s
1train/gradients/cross_entorpy/Sum_grad/Fill/valueConst*
value	B :*
dtype0*
_output_shapes
: 
ї
+train/gradients/cross_entorpy/Sum_grad/FillFill.train/gradients/cross_entorpy/Sum_grad/Shape_11train/gradients/cross_entorpy/Sum_grad/Fill/value*
T0*
_output_shapes
:
±
4train/gradients/cross_entorpy/Sum_grad/DynamicStitchDynamicStitch,train/gradients/cross_entorpy/Sum_grad/range*train/gradients/cross_entorpy/Sum_grad/mod,train/gradients/cross_entorpy/Sum_grad/Shape+train/gradients/cross_entorpy/Sum_grad/Fill*
T0*
N*#
_output_shapes
:€€€€€€€€€
r
0train/gradients/cross_entorpy/Sum_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
ѕ
.train/gradients/cross_entorpy/Sum_grad/MaximumMaximum4train/gradients/cross_entorpy/Sum_grad/DynamicStitch0train/gradients/cross_entorpy/Sum_grad/Maximum/y*#
_output_shapes
:€€€€€€€€€*
T0
Њ
/train/gradients/cross_entorpy/Sum_grad/floordivFloorDiv,train/gradients/cross_entorpy/Sum_grad/Shape.train/gradients/cross_entorpy/Sum_grad/Maximum*
_output_shapes
:*
T0
ћ
.train/gradients/cross_entorpy/Sum_grad/ReshapeReshape*train/gradients/cross_entorpy/Neg_grad/Neg4train/gradients/cross_entorpy/Sum_grad/DynamicStitch*
Tshape0*
_output_shapes
:*
T0
Ў
+train/gradients/cross_entorpy/Sum_grad/TileTile.train/gradients/cross_entorpy/Sum_grad/Reshape/train/gradients/cross_entorpy/Sum_grad/floordiv*

Tmultiples0*
T0*'
_output_shapes
:€€€€€€€€€
w
,train/gradients/cross_entorpy/mul_grad/ShapeShapeinputs/type*
T0*
out_type0*
_output_shapes
:

.train/gradients/cross_entorpy/mul_grad/Shape_1Shapecross_entorpy/Log*
T0*
out_type0*
_output_shapes
:
р
<train/gradients/cross_entorpy/mul_grad/BroadcastGradientArgsBroadcastGradientArgs,train/gradients/cross_entorpy/mul_grad/Shape.train/gradients/cross_entorpy/mul_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
£
*train/gradients/cross_entorpy/mul_grad/mulMul+train/gradients/cross_entorpy/Sum_grad/Tilecross_entorpy/Log*
T0*'
_output_shapes
:€€€€€€€€€
џ
*train/gradients/cross_entorpy/mul_grad/SumSum*train/gradients/cross_entorpy/mul_grad/mul<train/gradients/cross_entorpy/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
”
.train/gradients/cross_entorpy/mul_grad/ReshapeReshape*train/gradients/cross_entorpy/mul_grad/Sum,train/gradients/cross_entorpy/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
Я
,train/gradients/cross_entorpy/mul_grad/mul_1Mulinputs/type+train/gradients/cross_entorpy/Sum_grad/Tile*
T0*'
_output_shapes
:€€€€€€€€€
б
,train/gradients/cross_entorpy/mul_grad/Sum_1Sum,train/gradients/cross_entorpy/mul_grad/mul_1>train/gradients/cross_entorpy/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ў
0train/gradients/cross_entorpy/mul_grad/Reshape_1Reshape,train/gradients/cross_entorpy/mul_grad/Sum_1.train/gradients/cross_entorpy/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
£
7train/gradients/cross_entorpy/mul_grad/tuple/group_depsNoOp/^train/gradients/cross_entorpy/mul_grad/Reshape1^train/gradients/cross_entorpy/mul_grad/Reshape_1
™
?train/gradients/cross_entorpy/mul_grad/tuple/control_dependencyIdentity.train/gradients/cross_entorpy/mul_grad/Reshape8^train/gradients/cross_entorpy/mul_grad/tuple/group_deps*A
_class7
53loc:@train/gradients/cross_entorpy/mul_grad/Reshape*'
_output_shapes
:€€€€€€€€€*
T0
∞
Atrain/gradients/cross_entorpy/mul_grad/tuple/control_dependency_1Identity0train/gradients/cross_entorpy/mul_grad/Reshape_18^train/gradients/cross_entorpy/mul_grad/tuple/group_deps*
T0*C
_class9
75loc:@train/gradients/cross_entorpy/mul_grad/Reshape_1*'
_output_shapes
:€€€€€€€€€
∆
1train/gradients/cross_entorpy/Log_grad/Reciprocal
Reciprocaloutputs/SoftmaxB^train/gradients/cross_entorpy/mul_grad/tuple/control_dependency_1*'
_output_shapes
:€€€€€€€€€*
T0
ў
*train/gradients/cross_entorpy/Log_grad/mulMulAtrain/gradients/cross_entorpy/mul_grad/tuple/control_dependency_11train/gradients/cross_entorpy/Log_grad/Reciprocal*
T0*'
_output_shapes
:€€€€€€€€€
Ю
(train/gradients/outputs/Softmax_grad/mulMul*train/gradients/cross_entorpy/Log_grad/muloutputs/Softmax*
T0*'
_output_shapes
:€€€€€€€€€
Д
:train/gradients/outputs/Softmax_grad/Sum/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
а
(train/gradients/outputs/Softmax_grad/SumSum(train/gradients/outputs/Softmax_grad/mul:train/gradients/outputs/Softmax_grad/Sum/reduction_indices*#
_output_shapes
:€€€€€€€€€*
T0*
	keep_dims( *

Tidx0
Г
2train/gradients/outputs/Softmax_grad/Reshape/shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:
’
,train/gradients/outputs/Softmax_grad/ReshapeReshape(train/gradients/outputs/Softmax_grad/Sum2train/gradients/outputs/Softmax_grad/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
ї
(train/gradients/outputs/Softmax_grad/subSub*train/gradients/cross_entorpy/Log_grad/mul,train/gradients/outputs/Softmax_grad/Reshape*'
_output_shapes
:€€€€€€€€€*
T0
Ю
*train/gradients/outputs/Softmax_grad/mul_1Mul(train/gradients/outputs/Softmax_grad/suboutputs/Softmax*'
_output_shapes
:€€€€€€€€€*
T0
t
&train/gradients/outputs/add_grad/ShapeShapeoutputs/MatMul*
out_type0*
_output_shapes
:*
T0
r
(train/gradients/outputs/add_grad/Shape_1Const*
valueB:*
_output_shapes
:*
dtype0
ё
6train/gradients/outputs/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/outputs/add_grad/Shape(train/gradients/outputs/add_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
ѕ
$train/gradients/outputs/add_grad/SumSum*train/gradients/outputs/Softmax_grad/mul_16train/gradients/outputs/add_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ѕ
(train/gradients/outputs/add_grad/ReshapeReshape$train/gradients/outputs/add_grad/Sum&train/gradients/outputs/add_grad/Shape*
Tshape0*'
_output_shapes
:€€€€€€€€€*
T0
”
&train/gradients/outputs/add_grad/Sum_1Sum*train/gradients/outputs/Softmax_grad/mul_18train/gradients/outputs/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ї
*train/gradients/outputs/add_grad/Reshape_1Reshape&train/gradients/outputs/add_grad/Sum_1(train/gradients/outputs/add_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
С
1train/gradients/outputs/add_grad/tuple/group_depsNoOp)^train/gradients/outputs/add_grad/Reshape+^train/gradients/outputs/add_grad/Reshape_1
Т
9train/gradients/outputs/add_grad/tuple/control_dependencyIdentity(train/gradients/outputs/add_grad/Reshape2^train/gradients/outputs/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/outputs/add_grad/Reshape*'
_output_shapes
:€€€€€€€€€
Л
;train/gradients/outputs/add_grad/tuple/control_dependency_1Identity*train/gradients/outputs/add_grad/Reshape_12^train/gradients/outputs/add_grad/tuple/group_deps*=
_class3
1/loc:@train/gradients/outputs/add_grad/Reshape_1*
_output_shapes
:*
T0
а
*train/gradients/outputs/MatMul_grad/MatMulMatMul9train/gradients/outputs/add_grad/tuple/control_dependencyfc_2/weights/W_fc2/read*
transpose_b(*
T0*'
_output_shapes
:€€€€€€€€€d*
transpose_a( 
б
,train/gradients/outputs/MatMul_grad/MatMul_1MatMulfc_1/matmul_and_non_linear/Relu9train/gradients/outputs/add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:d*
transpose_a(
Ш
4train/gradients/outputs/MatMul_grad/tuple/group_depsNoOp+^train/gradients/outputs/MatMul_grad/MatMul-^train/gradients/outputs/MatMul_grad/MatMul_1
Ь
<train/gradients/outputs/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/outputs/MatMul_grad/MatMul5^train/gradients/outputs/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/outputs/MatMul_grad/MatMul*'
_output_shapes
:€€€€€€€€€d
Щ
>train/gradients/outputs/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/outputs/MatMul_grad/MatMul_15^train/gradients/outputs/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/outputs/MatMul_grad/MatMul_1*
_output_shapes

:d
Џ
=train/gradients/fc_1/matmul_and_non_linear/Relu_grad/ReluGradReluGrad<train/gradients/outputs/MatMul_grad/tuple/control_dependencyfc_1/matmul_and_non_linear/Relu*'
_output_shapes
:€€€€€€€€€d*
T0
Ъ
9train/gradients/fc_1/matmul_and_non_linear/add_grad/ShapeShape!fc_1/matmul_and_non_linear/MatMul*
T0*
out_type0*
_output_shapes
:
Е
;train/gradients/fc_1/matmul_and_non_linear/add_grad/Shape_1Const*
valueB:d*
_output_shapes
:*
dtype0
Ч
Itrain/gradients/fc_1/matmul_and_non_linear/add_grad/BroadcastGradientArgsBroadcastGradientArgs9train/gradients/fc_1/matmul_and_non_linear/add_grad/Shape;train/gradients/fc_1/matmul_and_non_linear/add_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
И
7train/gradients/fc_1/matmul_and_non_linear/add_grad/SumSum=train/gradients/fc_1/matmul_and_non_linear/Relu_grad/ReluGradItrain/gradients/fc_1/matmul_and_non_linear/add_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
ъ
;train/gradients/fc_1/matmul_and_non_linear/add_grad/ReshapeReshape7train/gradients/fc_1/matmul_and_non_linear/add_grad/Sum9train/gradients/fc_1/matmul_and_non_linear/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€d
М
9train/gradients/fc_1/matmul_and_non_linear/add_grad/Sum_1Sum=train/gradients/fc_1/matmul_and_non_linear/Relu_grad/ReluGradKtrain/gradients/fc_1/matmul_and_non_linear/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
у
=train/gradients/fc_1/matmul_and_non_linear/add_grad/Reshape_1Reshape9train/gradients/fc_1/matmul_and_non_linear/add_grad/Sum_1;train/gradients/fc_1/matmul_and_non_linear/add_grad/Shape_1*
Tshape0*
_output_shapes
:d*
T0
 
Dtrain/gradients/fc_1/matmul_and_non_linear/add_grad/tuple/group_depsNoOp<^train/gradients/fc_1/matmul_and_non_linear/add_grad/Reshape>^train/gradients/fc_1/matmul_and_non_linear/add_grad/Reshape_1
ё
Ltrain/gradients/fc_1/matmul_and_non_linear/add_grad/tuple/control_dependencyIdentity;train/gradients/fc_1/matmul_and_non_linear/add_grad/ReshapeE^train/gradients/fc_1/matmul_and_non_linear/add_grad/tuple/group_deps*N
_classD
B@loc:@train/gradients/fc_1/matmul_and_non_linear/add_grad/Reshape*'
_output_shapes
:€€€€€€€€€d*
T0
„
Ntrain/gradients/fc_1/matmul_and_non_linear/add_grad/tuple/control_dependency_1Identity=train/gradients/fc_1/matmul_and_non_linear/add_grad/Reshape_1E^train/gradients/fc_1/matmul_and_non_linear/add_grad/tuple/group_deps*
T0*P
_classF
DBloc:@train/gradients/fc_1/matmul_and_non_linear/add_grad/Reshape_1*
_output_shapes
:d
З
=train/gradients/fc_1/matmul_and_non_linear/MatMul_grad/MatMulMatMulLtrain/gradients/fc_1/matmul_and_non_linear/add_grad/tuple/control_dependencyfc_1/weights/W_fc1/read*
transpose_b(*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
T0
ш
?train/gradients/fc_1/matmul_and_non_linear/MatMul_grad/MatMul_1MatMulflatten/ReshapeLtrain/gradients/fc_1/matmul_and_non_linear/add_grad/tuple/control_dependency*
transpose_b( *
_output_shapes
:	Аd*
transpose_a(*
T0
—
Gtrain/gradients/fc_1/matmul_and_non_linear/MatMul_grad/tuple/group_depsNoOp>^train/gradients/fc_1/matmul_and_non_linear/MatMul_grad/MatMul@^train/gradients/fc_1/matmul_and_non_linear/MatMul_grad/MatMul_1
й
Otrain/gradients/fc_1/matmul_and_non_linear/MatMul_grad/tuple/control_dependencyIdentity=train/gradients/fc_1/matmul_and_non_linear/MatMul_grad/MatMulH^train/gradients/fc_1/matmul_and_non_linear/MatMul_grad/tuple/group_deps*P
_classF
DBloc:@train/gradients/fc_1/matmul_and_non_linear/MatMul_grad/MatMul*(
_output_shapes
:€€€€€€€€€А*
T0
ж
Qtrain/gradients/fc_1/matmul_and_non_linear/MatMul_grad/tuple/control_dependency_1Identity?train/gradients/fc_1/matmul_and_non_linear/MatMul_grad/MatMul_1H^train/gradients/fc_1/matmul_and_non_linear/MatMul_grad/tuple/group_deps*
T0*R
_classH
FDloc:@train/gradients/fc_1/matmul_and_non_linear/MatMul_grad/MatMul_1*
_output_shapes
:	Аd
}
*train/gradients/flatten/Reshape_grad/ShapeShapeconv_3/pool/MaxPool*
out_type0*
_output_shapes
:*
T0
ь
,train/gradients/flatten/Reshape_grad/ReshapeReshapeOtrain/gradients/fc_1/matmul_and_non_linear/MatMul_grad/tuple/control_dependency*train/gradients/flatten/Reshape_grad/Shape*
Tshape0*/
_output_shapes
:€€€€€€€€€(*
T0
Ї
4train/gradients/conv_3/pool/MaxPool_grad/MaxPoolGradMaxPoolGrad&conv_3/convolution_and_non_linear/Reluconv_3/pool/MaxPool,train/gradients/flatten/Reshape_grad/Reshape*
ksize
*/
_output_shapes
:€€€€€€€€€(*
T0*
data_formatNHWC*
strides
*
paddingSAME
и
Dtrain/gradients/conv_3/convolution_and_non_linear/Relu_grad/ReluGradReluGrad4train/gradients/conv_3/pool/MaxPool_grad/MaxPoolGrad&conv_3/convolution_and_non_linear/Relu*
T0*/
_output_shapes
:€€€€€€€€€(
®
@train/gradients/conv_3/convolution_and_non_linear/add_grad/ShapeShape(conv_3/convolution_and_non_linear/Conv2D*
T0*
out_type0*
_output_shapes
:
М
Btrain/gradients/conv_3/convolution_and_non_linear/add_grad/Shape_1Const*
valueB:(*
_output_shapes
:*
dtype0
ђ
Ptrain/gradients/conv_3/convolution_and_non_linear/add_grad/BroadcastGradientArgsBroadcastGradientArgs@train/gradients/conv_3/convolution_and_non_linear/add_grad/ShapeBtrain/gradients/conv_3/convolution_and_non_linear/add_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
Э
>train/gradients/conv_3/convolution_and_non_linear/add_grad/SumSumDtrain/gradients/conv_3/convolution_and_non_linear/Relu_grad/ReluGradPtrain/gradients/conv_3/convolution_and_non_linear/add_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ч
Btrain/gradients/conv_3/convolution_and_non_linear/add_grad/ReshapeReshape>train/gradients/conv_3/convolution_and_non_linear/add_grad/Sum@train/gradients/conv_3/convolution_and_non_linear/add_grad/Shape*
T0*
Tshape0*/
_output_shapes
:€€€€€€€€€(
°
@train/gradients/conv_3/convolution_and_non_linear/add_grad/Sum_1SumDtrain/gradients/conv_3/convolution_and_non_linear/Relu_grad/ReluGradRtrain/gradients/conv_3/convolution_and_non_linear/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
И
Dtrain/gradients/conv_3/convolution_and_non_linear/add_grad/Reshape_1Reshape@train/gradients/conv_3/convolution_and_non_linear/add_grad/Sum_1Btrain/gradients/conv_3/convolution_and_non_linear/add_grad/Shape_1*
Tshape0*
_output_shapes
:(*
T0
я
Ktrain/gradients/conv_3/convolution_and_non_linear/add_grad/tuple/group_depsNoOpC^train/gradients/conv_3/convolution_and_non_linear/add_grad/ReshapeE^train/gradients/conv_3/convolution_and_non_linear/add_grad/Reshape_1
В
Strain/gradients/conv_3/convolution_and_non_linear/add_grad/tuple/control_dependencyIdentityBtrain/gradients/conv_3/convolution_and_non_linear/add_grad/ReshapeL^train/gradients/conv_3/convolution_and_non_linear/add_grad/tuple/group_deps*U
_classK
IGloc:@train/gradients/conv_3/convolution_and_non_linear/add_grad/Reshape*/
_output_shapes
:€€€€€€€€€(*
T0
у
Utrain/gradients/conv_3/convolution_and_non_linear/add_grad/tuple/control_dependency_1IdentityDtrain/gradients/conv_3/convolution_and_non_linear/add_grad/Reshape_1L^train/gradients/conv_3/convolution_and_non_linear/add_grad/tuple/group_deps*
T0*W
_classM
KIloc:@train/gradients/conv_3/convolution_and_non_linear/add_grad/Reshape_1*
_output_shapes
:(
Ц
Ctrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/ShapeShapeconv_2/pool/MaxPool*
out_type0*
_output_shapes
:*
T0
ћ
Qtrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputCtrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/Shapeconv_3/weights/W_conv_3/readStrain/gradients/conv_3/convolution_and_non_linear/add_grad/tuple/control_dependency*
paddingSAME*
T0*
data_formatNHWC*
strides
*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
use_cudnn_on_gpu(
Ю
Etrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/Shape_1Const*%
valueB"         (   *
dtype0*
_output_shapes
:
£
Rtrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterconv_2/pool/MaxPoolEtrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/Shape_1Strain/gradients/conv_3/convolution_and_non_linear/add_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
T0*
paddingSAME*&
_output_shapes
:(*
data_formatNHWC*
strides

€
Ntrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/tuple/group_depsNoOpR^train/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInputS^train/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilter
¶
Vtrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/tuple/control_dependencyIdentityQtrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInputO^train/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/tuple/group_deps*d
_classZ
XVloc:@train/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:€€€€€€€€€*
T0
°
Xtrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/tuple/control_dependency_1IdentityRtrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilterO^train/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/tuple/group_deps*
T0*e
_class[
YWloc:@train/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:(
д
4train/gradients/conv_2/pool/MaxPool_grad/MaxPoolGradMaxPoolGrad&conv_2/convolution_and_non_linear/Reluconv_2/pool/MaxPoolVtrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/tuple/control_dependency*
ksize
*
T0*
paddingSAME*/
_output_shapes
:€€€€€€€€€  *
data_formatNHWC*
strides

и
Dtrain/gradients/conv_2/convolution_and_non_linear/Relu_grad/ReluGradReluGrad4train/gradients/conv_2/pool/MaxPool_grad/MaxPoolGrad&conv_2/convolution_and_non_linear/Relu*
T0*/
_output_shapes
:€€€€€€€€€  
®
@train/gradients/conv_2/convolution_and_non_linear/add_grad/ShapeShape(conv_2/convolution_and_non_linear/Conv2D*
out_type0*
_output_shapes
:*
T0
М
Btrain/gradients/conv_2/convolution_and_non_linear/add_grad/Shape_1Const*
valueB:*
_output_shapes
:*
dtype0
ђ
Ptrain/gradients/conv_2/convolution_and_non_linear/add_grad/BroadcastGradientArgsBroadcastGradientArgs@train/gradients/conv_2/convolution_and_non_linear/add_grad/ShapeBtrain/gradients/conv_2/convolution_and_non_linear/add_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
Э
>train/gradients/conv_2/convolution_and_non_linear/add_grad/SumSumDtrain/gradients/conv_2/convolution_and_non_linear/Relu_grad/ReluGradPtrain/gradients/conv_2/convolution_and_non_linear/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ч
Btrain/gradients/conv_2/convolution_and_non_linear/add_grad/ReshapeReshape>train/gradients/conv_2/convolution_and_non_linear/add_grad/Sum@train/gradients/conv_2/convolution_and_non_linear/add_grad/Shape*
Tshape0*/
_output_shapes
:€€€€€€€€€  *
T0
°
@train/gradients/conv_2/convolution_and_non_linear/add_grad/Sum_1SumDtrain/gradients/conv_2/convolution_and_non_linear/Relu_grad/ReluGradRtrain/gradients/conv_2/convolution_and_non_linear/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
И
Dtrain/gradients/conv_2/convolution_and_non_linear/add_grad/Reshape_1Reshape@train/gradients/conv_2/convolution_and_non_linear/add_grad/Sum_1Btrain/gradients/conv_2/convolution_and_non_linear/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
я
Ktrain/gradients/conv_2/convolution_and_non_linear/add_grad/tuple/group_depsNoOpC^train/gradients/conv_2/convolution_and_non_linear/add_grad/ReshapeE^train/gradients/conv_2/convolution_and_non_linear/add_grad/Reshape_1
В
Strain/gradients/conv_2/convolution_and_non_linear/add_grad/tuple/control_dependencyIdentityBtrain/gradients/conv_2/convolution_and_non_linear/add_grad/ReshapeL^train/gradients/conv_2/convolution_and_non_linear/add_grad/tuple/group_deps*
T0*U
_classK
IGloc:@train/gradients/conv_2/convolution_and_non_linear/add_grad/Reshape*/
_output_shapes
:€€€€€€€€€  
у
Utrain/gradients/conv_2/convolution_and_non_linear/add_grad/tuple/control_dependency_1IdentityDtrain/gradients/conv_2/convolution_and_non_linear/add_grad/Reshape_1L^train/gradients/conv_2/convolution_and_non_linear/add_grad/tuple/group_deps*W
_classM
KIloc:@train/gradients/conv_2/convolution_and_non_linear/add_grad/Reshape_1*
_output_shapes
:*
T0
Ц
Ctrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/ShapeShapeconv_1/pool/MaxPool*
out_type0*
_output_shapes
:*
T0
ћ
Qtrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputCtrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/Shapeconv_2/weights/W_conv_2/readStrain/gradients/conv_2/convolution_and_non_linear/add_grad/tuple/control_dependency*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingSAME
Ю
Etrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/Shape_1Const*%
valueB"      
      *
dtype0*
_output_shapes
:
£
Rtrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterconv_1/pool/MaxPoolEtrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/Shape_1Strain/gradients/conv_2/convolution_and_non_linear/add_grad/tuple/control_dependency*
paddingSAME*
T0*
data_formatNHWC*
strides
*&
_output_shapes
:
*
use_cudnn_on_gpu(
€
Ntrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/tuple/group_depsNoOpR^train/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInputS^train/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilter
¶
Vtrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/tuple/control_dependencyIdentityQtrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInputO^train/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@train/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:€€€€€€€€€  

°
Xtrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/tuple/control_dependency_1IdentityRtrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilterO^train/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/tuple/group_deps*
T0*e
_class[
YWloc:@train/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:

ж
4train/gradients/conv_1/pool/MaxPool_grad/MaxPoolGradMaxPoolGrad&conv_1/convolution_and_non_linear/Reluconv_1/pool/MaxPoolVtrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/tuple/control_dependency*
ksize
*1
_output_shapes
:€€€€€€€€€АА
*
T0*
data_formatNHWC*
strides
*
paddingSAME
к
Dtrain/gradients/conv_1/convolution_and_non_linear/Relu_grad/ReluGradReluGrad4train/gradients/conv_1/pool/MaxPool_grad/MaxPoolGrad&conv_1/convolution_and_non_linear/Relu*1
_output_shapes
:€€€€€€€€€АА
*
T0
®
@train/gradients/conv_1/convolution_and_non_linear/add_grad/ShapeShape(conv_1/convolution_and_non_linear/Conv2D*
T0*
out_type0*
_output_shapes
:
М
Btrain/gradients/conv_1/convolution_and_non_linear/add_grad/Shape_1Const*
valueB:
*
dtype0*
_output_shapes
:
ђ
Ptrain/gradients/conv_1/convolution_and_non_linear/add_grad/BroadcastGradientArgsBroadcastGradientArgs@train/gradients/conv_1/convolution_and_non_linear/add_grad/ShapeBtrain/gradients/conv_1/convolution_and_non_linear/add_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
Э
>train/gradients/conv_1/convolution_and_non_linear/add_grad/SumSumDtrain/gradients/conv_1/convolution_and_non_linear/Relu_grad/ReluGradPtrain/gradients/conv_1/convolution_and_non_linear/add_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Щ
Btrain/gradients/conv_1/convolution_and_non_linear/add_grad/ReshapeReshape>train/gradients/conv_1/convolution_and_non_linear/add_grad/Sum@train/gradients/conv_1/convolution_and_non_linear/add_grad/Shape*
T0*
Tshape0*1
_output_shapes
:€€€€€€€€€АА

°
@train/gradients/conv_1/convolution_and_non_linear/add_grad/Sum_1SumDtrain/gradients/conv_1/convolution_and_non_linear/Relu_grad/ReluGradRtrain/gradients/conv_1/convolution_and_non_linear/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
И
Dtrain/gradients/conv_1/convolution_and_non_linear/add_grad/Reshape_1Reshape@train/gradients/conv_1/convolution_and_non_linear/add_grad/Sum_1Btrain/gradients/conv_1/convolution_and_non_linear/add_grad/Shape_1*
Tshape0*
_output_shapes
:
*
T0
я
Ktrain/gradients/conv_1/convolution_and_non_linear/add_grad/tuple/group_depsNoOpC^train/gradients/conv_1/convolution_and_non_linear/add_grad/ReshapeE^train/gradients/conv_1/convolution_and_non_linear/add_grad/Reshape_1
Д
Strain/gradients/conv_1/convolution_and_non_linear/add_grad/tuple/control_dependencyIdentityBtrain/gradients/conv_1/convolution_and_non_linear/add_grad/ReshapeL^train/gradients/conv_1/convolution_and_non_linear/add_grad/tuple/group_deps*U
_classK
IGloc:@train/gradients/conv_1/convolution_and_non_linear/add_grad/Reshape*1
_output_shapes
:€€€€€€€€€АА
*
T0
у
Utrain/gradients/conv_1/convolution_and_non_linear/add_grad/tuple/control_dependency_1IdentityDtrain/gradients/conv_1/convolution_and_non_linear/add_grad/Reshape_1L^train/gradients/conv_1/convolution_and_non_linear/add_grad/tuple/group_deps*
T0*W
_classM
KIloc:@train/gradients/conv_1/convolution_and_non_linear/add_grad/Reshape_1*
_output_shapes
:

Ъ
Ctrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/ShapeShapeprepare_tensors/Reshape*
out_type0*
_output_shapes
:*
T0
ћ
Qtrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputCtrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/Shapeconv_1/weights/W_conv_1/readStrain/gradients/conv_1/convolution_and_non_linear/add_grad/tuple/control_dependency*
paddingSAME*
T0*
data_formatNHWC*
strides
*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
use_cudnn_on_gpu(
Ю
Etrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/Shape_1Const*%
valueB"         
   *
dtype0*
_output_shapes
:
І
Rtrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterprepare_tensors/ReshapeEtrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/Shape_1Strain/gradients/conv_1/convolution_and_non_linear/add_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
T0*
paddingSAME*&
_output_shapes
:
*
data_formatNHWC*
strides

€
Ntrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/tuple/group_depsNoOpR^train/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInputS^train/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilter
®
Vtrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/tuple/control_dependencyIdentityQtrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInputO^train/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@train/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInput*1
_output_shapes
:€€€€€€€€€АА
°
Xtrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/tuple/control_dependency_1IdentityRtrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilterO^train/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/tuple/group_deps*
T0*e
_class[
YWloc:@train/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:

Р
train/beta1_power/initial_valueConst*
valueB
 *fff?**
_class 
loc:@conv_1/weights/W_conv_1*
_output_shapes
: *
dtype0
°
train/beta1_power
VariableV2*
shape: *
_output_shapes
: *
shared_name **
_class 
loc:@conv_1/weights/W_conv_1*
dtype0*
	container 
ћ
train/beta1_power/AssignAssigntrain/beta1_powertrain/beta1_power/initial_value**
_class 
loc:@conv_1/weights/W_conv_1*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
В
train/beta1_power/readIdentitytrain/beta1_power**
_class 
loc:@conv_1/weights/W_conv_1*
_output_shapes
: *
T0
Р
train/beta2_power/initial_valueConst*
valueB
 *wЊ?**
_class 
loc:@conv_1/weights/W_conv_1*
_output_shapes
: *
dtype0
°
train/beta2_power
VariableV2*
	container *
dtype0**
_class 
loc:@conv_1/weights/W_conv_1*
_output_shapes
: *
shape: *
shared_name 
ћ
train/beta2_power/AssignAssigntrain/beta2_powertrain/beta2_power/initial_value*
use_locking(*
T0**
_class 
loc:@conv_1/weights/W_conv_1*
validate_shape(*
_output_shapes
: 
В
train/beta2_power/readIdentitytrain/beta2_power**
_class 
loc:@conv_1/weights/W_conv_1*
_output_shapes
: *
T0
p
train/zerosConst*%
valueB
*    *
dtype0*&
_output_shapes
:

ћ
conv_1/weights/W_conv_1/Adam
VariableV2*
shared_name **
_class 
loc:@conv_1/weights/W_conv_1*
	container *
shape:
*
dtype0*&
_output_shapes
:

ё
#conv_1/weights/W_conv_1/Adam/AssignAssignconv_1/weights/W_conv_1/Adamtrain/zeros*
use_locking(*
T0**
_class 
loc:@conv_1/weights/W_conv_1*
validate_shape(*&
_output_shapes
:

®
!conv_1/weights/W_conv_1/Adam/readIdentityconv_1/weights/W_conv_1/Adam**
_class 
loc:@conv_1/weights/W_conv_1*&
_output_shapes
:
*
T0
r
train/zeros_1Const*%
valueB
*    *&
_output_shapes
:
*
dtype0
ќ
conv_1/weights/W_conv_1/Adam_1
VariableV2*
shape:
*&
_output_shapes
:
*
shared_name **
_class 
loc:@conv_1/weights/W_conv_1*
dtype0*
	container 
д
%conv_1/weights/W_conv_1/Adam_1/AssignAssignconv_1/weights/W_conv_1/Adam_1train/zeros_1*
use_locking(*
T0**
_class 
loc:@conv_1/weights/W_conv_1*
validate_shape(*&
_output_shapes
:

ђ
#conv_1/weights/W_conv_1/Adam_1/readIdentityconv_1/weights/W_conv_1/Adam_1**
_class 
loc:@conv_1/weights/W_conv_1*&
_output_shapes
:
*
T0
Z
train/zeros_2Const*
valueB
*    *
dtype0*
_output_shapes
:

≤
conv_1/biases/b_conv_1/Adam
VariableV2*)
_class
loc:@conv_1/biases/b_conv_1*
_output_shapes
:
*
shape:
*
dtype0*
shared_name *
	container 
—
"conv_1/biases/b_conv_1/Adam/AssignAssignconv_1/biases/b_conv_1/Adamtrain/zeros_2*
use_locking(*
T0*)
_class
loc:@conv_1/biases/b_conv_1*
validate_shape(*
_output_shapes
:

Щ
 conv_1/biases/b_conv_1/Adam/readIdentityconv_1/biases/b_conv_1/Adam*
T0*)
_class
loc:@conv_1/biases/b_conv_1*
_output_shapes
:

Z
train/zeros_3Const*
valueB
*    *
dtype0*
_output_shapes
:

і
conv_1/biases/b_conv_1/Adam_1
VariableV2*)
_class
loc:@conv_1/biases/b_conv_1*
_output_shapes
:
*
shape:
*
dtype0*
shared_name *
	container 
’
$conv_1/biases/b_conv_1/Adam_1/AssignAssignconv_1/biases/b_conv_1/Adam_1train/zeros_3*)
_class
loc:@conv_1/biases/b_conv_1*
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
Э
"conv_1/biases/b_conv_1/Adam_1/readIdentityconv_1/biases/b_conv_1/Adam_1*
T0*)
_class
loc:@conv_1/biases/b_conv_1*
_output_shapes
:

r
train/zeros_4Const*%
valueB
*    *&
_output_shapes
:
*
dtype0
ћ
conv_2/weights/W_conv_2/Adam
VariableV2*
	container *
dtype0**
_class 
loc:@conv_2/weights/W_conv_2*&
_output_shapes
:
*
shape:
*
shared_name 
а
#conv_2/weights/W_conv_2/Adam/AssignAssignconv_2/weights/W_conv_2/Adamtrain/zeros_4*
use_locking(*
T0**
_class 
loc:@conv_2/weights/W_conv_2*
validate_shape(*&
_output_shapes
:

®
!conv_2/weights/W_conv_2/Adam/readIdentityconv_2/weights/W_conv_2/Adam**
_class 
loc:@conv_2/weights/W_conv_2*&
_output_shapes
:
*
T0
r
train/zeros_5Const*%
valueB
*    *&
_output_shapes
:
*
dtype0
ќ
conv_2/weights/W_conv_2/Adam_1
VariableV2*
	container *
dtype0**
_class 
loc:@conv_2/weights/W_conv_2*&
_output_shapes
:
*
shape:
*
shared_name 
д
%conv_2/weights/W_conv_2/Adam_1/AssignAssignconv_2/weights/W_conv_2/Adam_1train/zeros_5**
_class 
loc:@conv_2/weights/W_conv_2*&
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
ђ
#conv_2/weights/W_conv_2/Adam_1/readIdentityconv_2/weights/W_conv_2/Adam_1**
_class 
loc:@conv_2/weights/W_conv_2*&
_output_shapes
:
*
T0
Z
train/zeros_6Const*
valueB*    *
_output_shapes
:*
dtype0
≤
conv_2/biases/b_conv_2/Adam
VariableV2*
shared_name *)
_class
loc:@conv_2/biases/b_conv_2*
	container *
shape:*
dtype0*
_output_shapes
:
—
"conv_2/biases/b_conv_2/Adam/AssignAssignconv_2/biases/b_conv_2/Adamtrain/zeros_6*
use_locking(*
T0*)
_class
loc:@conv_2/biases/b_conv_2*
validate_shape(*
_output_shapes
:
Щ
 conv_2/biases/b_conv_2/Adam/readIdentityconv_2/biases/b_conv_2/Adam*
T0*)
_class
loc:@conv_2/biases/b_conv_2*
_output_shapes
:
Z
train/zeros_7Const*
valueB*    *
_output_shapes
:*
dtype0
і
conv_2/biases/b_conv_2/Adam_1
VariableV2*
shape:*
_output_shapes
:*
shared_name *)
_class
loc:@conv_2/biases/b_conv_2*
dtype0*
	container 
’
$conv_2/biases/b_conv_2/Adam_1/AssignAssignconv_2/biases/b_conv_2/Adam_1train/zeros_7*)
_class
loc:@conv_2/biases/b_conv_2*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
Э
"conv_2/biases/b_conv_2/Adam_1/readIdentityconv_2/biases/b_conv_2/Adam_1*
T0*)
_class
loc:@conv_2/biases/b_conv_2*
_output_shapes
:
r
train/zeros_8Const*%
valueB(*    *&
_output_shapes
:(*
dtype0
ћ
conv_3/weights/W_conv_3/Adam
VariableV2*
shape:(*&
_output_shapes
:(*
shared_name **
_class 
loc:@conv_3/weights/W_conv_3*
dtype0*
	container 
а
#conv_3/weights/W_conv_3/Adam/AssignAssignconv_3/weights/W_conv_3/Adamtrain/zeros_8**
_class 
loc:@conv_3/weights/W_conv_3*&
_output_shapes
:(*
T0*
validate_shape(*
use_locking(
®
!conv_3/weights/W_conv_3/Adam/readIdentityconv_3/weights/W_conv_3/Adam*
T0**
_class 
loc:@conv_3/weights/W_conv_3*&
_output_shapes
:(
r
train/zeros_9Const*%
valueB(*    *&
_output_shapes
:(*
dtype0
ќ
conv_3/weights/W_conv_3/Adam_1
VariableV2**
_class 
loc:@conv_3/weights/W_conv_3*&
_output_shapes
:(*
shape:(*
dtype0*
shared_name *
	container 
д
%conv_3/weights/W_conv_3/Adam_1/AssignAssignconv_3/weights/W_conv_3/Adam_1train/zeros_9*
use_locking(*
T0**
_class 
loc:@conv_3/weights/W_conv_3*
validate_shape(*&
_output_shapes
:(
ђ
#conv_3/weights/W_conv_3/Adam_1/readIdentityconv_3/weights/W_conv_3/Adam_1**
_class 
loc:@conv_3/weights/W_conv_3*&
_output_shapes
:(*
T0
[
train/zeros_10Const*
valueB(*    *
_output_shapes
:(*
dtype0
≤
conv_3/biases/b_conv_3/Adam
VariableV2*
shared_name *)
_class
loc:@conv_3/biases/b_conv_3*
	container *
shape:(*
dtype0*
_output_shapes
:(
“
"conv_3/biases/b_conv_3/Adam/AssignAssignconv_3/biases/b_conv_3/Adamtrain/zeros_10*)
_class
loc:@conv_3/biases/b_conv_3*
_output_shapes
:(*
T0*
validate_shape(*
use_locking(
Щ
 conv_3/biases/b_conv_3/Adam/readIdentityconv_3/biases/b_conv_3/Adam*)
_class
loc:@conv_3/biases/b_conv_3*
_output_shapes
:(*
T0
[
train/zeros_11Const*
valueB(*    *
dtype0*
_output_shapes
:(
і
conv_3/biases/b_conv_3/Adam_1
VariableV2*)
_class
loc:@conv_3/biases/b_conv_3*
_output_shapes
:(*
shape:(*
dtype0*
shared_name *
	container 
÷
$conv_3/biases/b_conv_3/Adam_1/AssignAssignconv_3/biases/b_conv_3/Adam_1train/zeros_11*
use_locking(*
T0*)
_class
loc:@conv_3/biases/b_conv_3*
validate_shape(*
_output_shapes
:(
Э
"conv_3/biases/b_conv_3/Adam_1/readIdentityconv_3/biases/b_conv_3/Adam_1*)
_class
loc:@conv_3/biases/b_conv_3*
_output_shapes
:(*
T0
e
train/zeros_12Const*
valueB	Аd*    *
dtype0*
_output_shapes
:	Аd
і
fc_1/weights/W_fc1/Adam
VariableV2*
shape:	Аd*
_output_shapes
:	Аd*
shared_name *%
_class
loc:@fc_1/weights/W_fc1*
dtype0*
	container 
Ћ
fc_1/weights/W_fc1/Adam/AssignAssignfc_1/weights/W_fc1/Adamtrain/zeros_12*%
_class
loc:@fc_1/weights/W_fc1*
_output_shapes
:	Аd*
T0*
validate_shape(*
use_locking(
Т
fc_1/weights/W_fc1/Adam/readIdentityfc_1/weights/W_fc1/Adam*%
_class
loc:@fc_1/weights/W_fc1*
_output_shapes
:	Аd*
T0
e
train/zeros_13Const*
valueB	Аd*    *
_output_shapes
:	Аd*
dtype0
ґ
fc_1/weights/W_fc1/Adam_1
VariableV2*
shared_name *%
_class
loc:@fc_1/weights/W_fc1*
	container *
shape:	Аd*
dtype0*
_output_shapes
:	Аd
ѕ
 fc_1/weights/W_fc1/Adam_1/AssignAssignfc_1/weights/W_fc1/Adam_1train/zeros_13*%
_class
loc:@fc_1/weights/W_fc1*
_output_shapes
:	Аd*
T0*
validate_shape(*
use_locking(
Ц
fc_1/weights/W_fc1/Adam_1/readIdentityfc_1/weights/W_fc1/Adam_1*%
_class
loc:@fc_1/weights/W_fc1*
_output_shapes
:	Аd*
T0
[
train/zeros_14Const*
valueBd*    *
dtype0*
_output_shapes
:d
®
fc_1/biases/b_fc1/Adam
VariableV2*$
_class
loc:@fc_1/biases/b_fc1*
_output_shapes
:d*
shape:d*
dtype0*
shared_name *
	container 
√
fc_1/biases/b_fc1/Adam/AssignAssignfc_1/biases/b_fc1/Adamtrain/zeros_14*
use_locking(*
T0*$
_class
loc:@fc_1/biases/b_fc1*
validate_shape(*
_output_shapes
:d
К
fc_1/biases/b_fc1/Adam/readIdentityfc_1/biases/b_fc1/Adam*$
_class
loc:@fc_1/biases/b_fc1*
_output_shapes
:d*
T0
[
train/zeros_15Const*
valueBd*    *
_output_shapes
:d*
dtype0
™
fc_1/biases/b_fc1/Adam_1
VariableV2*
	container *
dtype0*$
_class
loc:@fc_1/biases/b_fc1*
_output_shapes
:d*
shape:d*
shared_name 
«
fc_1/biases/b_fc1/Adam_1/AssignAssignfc_1/biases/b_fc1/Adam_1train/zeros_15*
use_locking(*
T0*$
_class
loc:@fc_1/biases/b_fc1*
validate_shape(*
_output_shapes
:d
О
fc_1/biases/b_fc1/Adam_1/readIdentityfc_1/biases/b_fc1/Adam_1*$
_class
loc:@fc_1/biases/b_fc1*
_output_shapes
:d*
T0
c
train/zeros_16Const*
valueBd*    *
dtype0*
_output_shapes

:d
≤
fc_2/weights/W_fc2/Adam
VariableV2*%
_class
loc:@fc_2/weights/W_fc2*
_output_shapes

:d*
shape
:d*
dtype0*
shared_name *
	container 
 
fc_2/weights/W_fc2/Adam/AssignAssignfc_2/weights/W_fc2/Adamtrain/zeros_16*
use_locking(*
T0*%
_class
loc:@fc_2/weights/W_fc2*
validate_shape(*
_output_shapes

:d
С
fc_2/weights/W_fc2/Adam/readIdentityfc_2/weights/W_fc2/Adam*%
_class
loc:@fc_2/weights/W_fc2*
_output_shapes

:d*
T0
c
train/zeros_17Const*
valueBd*    *
dtype0*
_output_shapes

:d
і
fc_2/weights/W_fc2/Adam_1
VariableV2*%
_class
loc:@fc_2/weights/W_fc2*
_output_shapes

:d*
shape
:d*
dtype0*
shared_name *
	container 
ќ
 fc_2/weights/W_fc2/Adam_1/AssignAssignfc_2/weights/W_fc2/Adam_1train/zeros_17*%
_class
loc:@fc_2/weights/W_fc2*
_output_shapes

:d*
T0*
validate_shape(*
use_locking(
Х
fc_2/weights/W_fc2/Adam_1/readIdentityfc_2/weights/W_fc2/Adam_1*%
_class
loc:@fc_2/weights/W_fc2*
_output_shapes

:d*
T0
[
train/zeros_18Const*
valueB*    *
dtype0*
_output_shapes
:
™
fc_2/biases/b_fc_2/Adam
VariableV2*
	container *
dtype0*%
_class
loc:@fc_2/biases/b_fc_2*
_output_shapes
:*
shape:*
shared_name 
∆
fc_2/biases/b_fc_2/Adam/AssignAssignfc_2/biases/b_fc_2/Adamtrain/zeros_18*%
_class
loc:@fc_2/biases/b_fc_2*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
Н
fc_2/biases/b_fc_2/Adam/readIdentityfc_2/biases/b_fc_2/Adam*%
_class
loc:@fc_2/biases/b_fc_2*
_output_shapes
:*
T0
[
train/zeros_19Const*
valueB*    *
dtype0*
_output_shapes
:
ђ
fc_2/biases/b_fc_2/Adam_1
VariableV2*
shared_name *%
_class
loc:@fc_2/biases/b_fc_2*
	container *
shape:*
dtype0*
_output_shapes
:
 
 fc_2/biases/b_fc_2/Adam_1/AssignAssignfc_2/biases/b_fc_2/Adam_1train/zeros_19*%
_class
loc:@fc_2/biases/b_fc_2*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
С
fc_2/biases/b_fc_2/Adam_1/readIdentityfc_2/biases/b_fc_2/Adam_1*%
_class
loc:@fc_2/biases/b_fc_2*
_output_shapes
:*
T0
]
train/Adam/learning_rateConst*
valueB
 *Ј—8*
_output_shapes
: *
dtype0
U
train/Adam/beta1Const*
valueB
 *fff?*
_output_shapes
: *
dtype0
U
train/Adam/beta2Const*
valueB
 *wЊ?*
_output_shapes
: *
dtype0
W
train/Adam/epsilonConst*
valueB
 *wћ+2*
_output_shapes
: *
dtype0
г
3train/Adam/update_conv_1/weights/W_conv_1/ApplyAdam	ApplyAdamconv_1/weights/W_conv_1conv_1/weights/W_conv_1/Adamconv_1/weights/W_conv_1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonXtrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0**
_class 
loc:@conv_1/weights/W_conv_1*&
_output_shapes
:

ѕ
2train/Adam/update_conv_1/biases/b_conv_1/ApplyAdam	ApplyAdamconv_1/biases/b_conv_1conv_1/biases/b_conv_1/Adamconv_1/biases/b_conv_1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonUtrain/gradients/conv_1/convolution_and_non_linear/add_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@conv_1/biases/b_conv_1*
_output_shapes
:

г
3train/Adam/update_conv_2/weights/W_conv_2/ApplyAdam	ApplyAdamconv_2/weights/W_conv_2conv_2/weights/W_conv_2/Adamconv_2/weights/W_conv_2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonXtrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0**
_class 
loc:@conv_2/weights/W_conv_2*&
_output_shapes
:

ѕ
2train/Adam/update_conv_2/biases/b_conv_2/ApplyAdam	ApplyAdamconv_2/biases/b_conv_2conv_2/biases/b_conv_2/Adamconv_2/biases/b_conv_2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonUtrain/gradients/conv_2/convolution_and_non_linear/add_grad/tuple/control_dependency_1*)
_class
loc:@conv_2/biases/b_conv_2*
_output_shapes
:*
T0*
use_locking( 
г
3train/Adam/update_conv_3/weights/W_conv_3/ApplyAdam	ApplyAdamconv_3/weights/W_conv_3conv_3/weights/W_conv_3/Adamconv_3/weights/W_conv_3/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonXtrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0**
_class 
loc:@conv_3/weights/W_conv_3*&
_output_shapes
:(
ѕ
2train/Adam/update_conv_3/biases/b_conv_3/ApplyAdam	ApplyAdamconv_3/biases/b_conv_3conv_3/biases/b_conv_3/Adamconv_3/biases/b_conv_3/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonUtrain/gradients/conv_3/convolution_and_non_linear/add_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@conv_3/biases/b_conv_3*
_output_shapes
:(
Љ
.train/Adam/update_fc_1/weights/W_fc1/ApplyAdam	ApplyAdamfc_1/weights/W_fc1fc_1/weights/W_fc1/Adamfc_1/weights/W_fc1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonQtrain/gradients/fc_1/matmul_and_non_linear/MatMul_grad/tuple/control_dependency_1*%
_class
loc:@fc_1/weights/W_fc1*
_output_shapes
:	Аd*
T0*
use_locking( 
ѓ
-train/Adam/update_fc_1/biases/b_fc1/ApplyAdam	ApplyAdamfc_1/biases/b_fc1fc_1/biases/b_fc1/Adamfc_1/biases/b_fc1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonNtrain/gradients/fc_1/matmul_and_non_linear/add_grad/tuple/control_dependency_1*
use_locking( *
T0*$
_class
loc:@fc_1/biases/b_fc1*
_output_shapes
:d
®
.train/Adam/update_fc_2/weights/W_fc2/ApplyAdam	ApplyAdamfc_2/weights/W_fc2fc_2/weights/W_fc2/Adamfc_2/weights/W_fc2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/outputs/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*%
_class
loc:@fc_2/weights/W_fc2*
_output_shapes

:d
°
.train/Adam/update_fc_2/biases/b_fc_2/ApplyAdam	ApplyAdamfc_2/biases/b_fc_2fc_2/biases/b_fc_2/Adamfc_2/biases/b_fc_2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/outputs/add_grad/tuple/control_dependency_1*%
_class
loc:@fc_2/biases/b_fc_2*
_output_shapes
:*
T0*
use_locking( 
Р
train/Adam/mulMultrain/beta1_power/readtrain/Adam/beta14^train/Adam/update_conv_1/weights/W_conv_1/ApplyAdam3^train/Adam/update_conv_1/biases/b_conv_1/ApplyAdam4^train/Adam/update_conv_2/weights/W_conv_2/ApplyAdam3^train/Adam/update_conv_2/biases/b_conv_2/ApplyAdam4^train/Adam/update_conv_3/weights/W_conv_3/ApplyAdam3^train/Adam/update_conv_3/biases/b_conv_3/ApplyAdam/^train/Adam/update_fc_1/weights/W_fc1/ApplyAdam.^train/Adam/update_fc_1/biases/b_fc1/ApplyAdam/^train/Adam/update_fc_2/weights/W_fc2/ApplyAdam/^train/Adam/update_fc_2/biases/b_fc_2/ApplyAdam**
_class 
loc:@conv_1/weights/W_conv_1*
_output_shapes
: *
T0
і
train/Adam/AssignAssigntrain/beta1_powertrain/Adam/mul*
use_locking( *
T0**
_class 
loc:@conv_1/weights/W_conv_1*
validate_shape(*
_output_shapes
: 
Т
train/Adam/mul_1Multrain/beta2_power/readtrain/Adam/beta24^train/Adam/update_conv_1/weights/W_conv_1/ApplyAdam3^train/Adam/update_conv_1/biases/b_conv_1/ApplyAdam4^train/Adam/update_conv_2/weights/W_conv_2/ApplyAdam3^train/Adam/update_conv_2/biases/b_conv_2/ApplyAdam4^train/Adam/update_conv_3/weights/W_conv_3/ApplyAdam3^train/Adam/update_conv_3/biases/b_conv_3/ApplyAdam/^train/Adam/update_fc_1/weights/W_fc1/ApplyAdam.^train/Adam/update_fc_1/biases/b_fc1/ApplyAdam/^train/Adam/update_fc_2/weights/W_fc2/ApplyAdam/^train/Adam/update_fc_2/biases/b_fc_2/ApplyAdam**
_class 
loc:@conv_1/weights/W_conv_1*
_output_shapes
: *
T0
Є
train/Adam/Assign_1Assigntrain/beta2_powertrain/Adam/mul_1**
_class 
loc:@conv_1/weights/W_conv_1*
_output_shapes
: *
T0*
validate_shape(*
use_locking( 
ј

train/AdamNoOp4^train/Adam/update_conv_1/weights/W_conv_1/ApplyAdam3^train/Adam/update_conv_1/biases/b_conv_1/ApplyAdam4^train/Adam/update_conv_2/weights/W_conv_2/ApplyAdam3^train/Adam/update_conv_2/biases/b_conv_2/ApplyAdam4^train/Adam/update_conv_3/weights/W_conv_3/ApplyAdam3^train/Adam/update_conv_3/biases/b_conv_3/ApplyAdam/^train/Adam/update_fc_1/weights/W_fc1/ApplyAdam.^train/Adam/update_fc_1/biases/b_fc1/ApplyAdam/^train/Adam/update_fc_2/weights/W_fc2/ApplyAdam/^train/Adam/update_fc_2/biases/b_fc_2/ApplyAdam^train/Adam/Assign^train/Adam/Assign_1
[
evaluate/ArgMax/dimensionConst*
value	B :*
_output_shapes
: *
dtype0

evaluate/ArgMaxArgMaxoutputs/Softmaxevaluate/ArgMax/dimension*#
_output_shapes
:€€€€€€€€€*
T0*

Tidx0
]
evaluate/ArgMax_1/dimensionConst*
value	B :*
_output_shapes
: *
dtype0

evaluate/ArgMax_1ArgMaxinputs/typeevaluate/ArgMax_1/dimension*#
_output_shapes
:€€€€€€€€€*
T0*

Tidx0
i
evaluate/EqualEqualevaluate/ArgMaxevaluate/ArgMax_1*#
_output_shapes
:€€€€€€€€€*
T0	
b
evaluate/CastCastevaluate/Equal*#
_output_shapes
:€€€€€€€€€*

DstT0*

SrcT0

X
evaluate/ConstConst*
valueB: *
dtype0*
_output_shapes
:
r
evaluate/MeanMeanevaluate/Castevaluate/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
`
Merge/MergeSummaryMergeSummarycross_entorpy/cross_entorpy*
N*
_output_shapes
: "ѓVЇз8     ≤Х£b	e°•D÷AJЏс
№ Є 
9
Add
x"T
y"T
z"T"
Ttype:
2	
—
	ApplyAdam
var"TА	
m"TА	
v"TА
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"TА"
Ttype:
2	"
use_lockingbool( 
l
ArgMax

input"T
	dimension"Tidx

output	"
Ttype:
2	"
Tidxtype0:
2	
x
Assign
ref"TА

value"T

output_ref"TА"	
Ttype"
validate_shapebool("
use_lockingbool(Ш
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
8
Const
output"dtype"
valuetensor"
dtypetype
…
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
п
Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
о
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
A
Equal
x"T
y"T
z
"
Ttype:
2	
Р
4
Fill
dims

value"T
output"T"	
Ttype
>
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
7
FloorMod
x"T
y"T
z"T"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
+
Log
x"T
y"T"
Ttype:	
2
o
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2
Њ
MaxPool

input"T
output"T"
Ttype0:
2"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
д
MaxPoolGrad

orig_input"T
orig_output"T	
grad"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype0:
2
:
Maximum
x"T
y"T
z"T"
Ttype:	
2	Р
К
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
8
MergeSummary
inputs*N
summary"
Nint(0
<
Mul
x"T
y"T
z"T"
Ttype:
2	Р
-
Neg
x"T
y"T"
Ttype:
	2	

NoOp
A
Placeholder
output"dtype"
dtypetype"
shapeshape: 
К
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
`
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2	
=
RealDiv
x"T
y"T
z"T"
Ttype:
2	
4

Reciprocal
x"T
y"T"
Ttype:
	2	
A
Relu
features"T
activations"T"
Ttype:
2		
S
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2		
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
M
ScalarSummary
tags
values"T
summary"
Ttype:
2		
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
8
Softmax
logits"T
softmax"T"
Ttype:
2
5
Sub
x"T
y"T
z"T"
Ttype:
	2	
Й
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	

TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
s

VariableV2
ref"dtypeА"
shapeshape"
dtypetype"
	containerstring "
shared_namestring И*1.0.12v1.0.0-65-g4763edf-dirtyУ±
a
inputs/graphsPlaceholder*
shape: *
dtype0*)
_output_shapes
:€€€€€€€€€АА
]
inputs/typePlaceholder*
shape: *
dtype0*'
_output_shapes
:€€€€€€€€€
v
prepare_tensors/Reshape/shapeConst*%
valueB"€€€€А   А      *
_output_shapes
:*
dtype0
Ъ
prepare_tensors/ReshapeReshapeinputs/graphsprepare_tensors/Reshape/shape*
T0*
Tshape0*1
_output_shapes
:€€€€€€€€€АА
~
%conv_1/weights/truncated_normal/shapeConst*%
valueB"         
   *
dtype0*
_output_shapes
:
i
$conv_1/weights/truncated_normal/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
k
&conv_1/weights/truncated_normal/stddevConst*
valueB
 *Ќћћ=*
_output_shapes
: *
dtype0
ј
/conv_1/weights/truncated_normal/TruncatedNormalTruncatedNormal%conv_1/weights/truncated_normal/shape*&
_output_shapes
:
*
seed2 *
T0*

seed *
dtype0
і
#conv_1/weights/truncated_normal/mulMul/conv_1/weights/truncated_normal/TruncatedNormal&conv_1/weights/truncated_normal/stddev*
T0*&
_output_shapes
:

Ґ
conv_1/weights/truncated_normalAdd#conv_1/weights/truncated_normal/mul$conv_1/weights/truncated_normal/mean*&
_output_shapes
:
*
T0
Ы
conv_1/weights/W_conv_1
VariableV2*
shape:
*
shared_name *
dtype0*&
_output_shapes
:
*
	container 
и
conv_1/weights/W_conv_1/AssignAssignconv_1/weights/W_conv_1conv_1/weights/truncated_normal*
use_locking(*
T0**
_class 
loc:@conv_1/weights/W_conv_1*
validate_shape(*&
_output_shapes
:

Ю
conv_1/weights/W_conv_1/readIdentityconv_1/weights/W_conv_1**
_class 
loc:@conv_1/weights/W_conv_1*&
_output_shapes
:
*
T0
`
conv_1/biases/ConstConst*
valueB
*Ќћћ=*
dtype0*
_output_shapes
:

В
conv_1/biases/b_conv_1
VariableV2*
_output_shapes
:
*
	container *
shape:
*
dtype0*
shared_name 
Ќ
conv_1/biases/b_conv_1/AssignAssignconv_1/biases/b_conv_1conv_1/biases/Const*
use_locking(*
T0*)
_class
loc:@conv_1/biases/b_conv_1*
validate_shape(*
_output_shapes
:

П
conv_1/biases/b_conv_1/readIdentityconv_1/biases/b_conv_1*
T0*)
_class
loc:@conv_1/biases/b_conv_1*
_output_shapes
:

ь
(conv_1/convolution_and_non_linear/Conv2DConv2Dprepare_tensors/Reshapeconv_1/weights/W_conv_1/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:€€€€€€€€€АА

ѓ
%conv_1/convolution_and_non_linear/addAdd(conv_1/convolution_and_non_linear/Conv2Dconv_1/biases/b_conv_1/read*
T0*1
_output_shapes
:€€€€€€€€€АА

С
&conv_1/convolution_and_non_linear/ReluRelu%conv_1/convolution_and_non_linear/add*
T0*1
_output_shapes
:€€€€€€€€€АА

“
conv_1/pool/MaxPoolMaxPool&conv_1/convolution_and_non_linear/Relu*
paddingSAME*
T0*
strides
*
data_formatNHWC*/
_output_shapes
:€€€€€€€€€  
*
ksize

~
%conv_2/weights/truncated_normal/shapeConst*%
valueB"      
      *
_output_shapes
:*
dtype0
i
$conv_2/weights/truncated_normal/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
k
&conv_2/weights/truncated_normal/stddevConst*
valueB
 *Ќћћ=*
dtype0*
_output_shapes
: 
ј
/conv_2/weights/truncated_normal/TruncatedNormalTruncatedNormal%conv_2/weights/truncated_normal/shape*

seed *
T0*
dtype0*&
_output_shapes
:
*
seed2 
і
#conv_2/weights/truncated_normal/mulMul/conv_2/weights/truncated_normal/TruncatedNormal&conv_2/weights/truncated_normal/stddev*&
_output_shapes
:
*
T0
Ґ
conv_2/weights/truncated_normalAdd#conv_2/weights/truncated_normal/mul$conv_2/weights/truncated_normal/mean*&
_output_shapes
:
*
T0
Ы
conv_2/weights/W_conv_2
VariableV2*&
_output_shapes
:
*
	container *
shape:
*
dtype0*
shared_name 
и
conv_2/weights/W_conv_2/AssignAssignconv_2/weights/W_conv_2conv_2/weights/truncated_normal**
_class 
loc:@conv_2/weights/W_conv_2*&
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
Ю
conv_2/weights/W_conv_2/readIdentityconv_2/weights/W_conv_2**
_class 
loc:@conv_2/weights/W_conv_2*&
_output_shapes
:
*
T0
`
conv_2/biases/ConstConst*
valueB*Ќћћ=*
dtype0*
_output_shapes
:
В
conv_2/biases/b_conv_2
VariableV2*
_output_shapes
:*
	container *
shape:*
dtype0*
shared_name 
Ќ
conv_2/biases/b_conv_2/AssignAssignconv_2/biases/b_conv_2conv_2/biases/Const*
use_locking(*
T0*)
_class
loc:@conv_2/biases/b_conv_2*
validate_shape(*
_output_shapes
:
П
conv_2/biases/b_conv_2/readIdentityconv_2/biases/b_conv_2*
T0*)
_class
loc:@conv_2/biases/b_conv_2*
_output_shapes
:
ц
(conv_2/convolution_and_non_linear/Conv2DConv2Dconv_1/pool/MaxPoolconv_2/weights/W_conv_2/read*/
_output_shapes
:€€€€€€€€€  *
T0*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC*
paddingSAME
≠
%conv_2/convolution_and_non_linear/addAdd(conv_2/convolution_and_non_linear/Conv2Dconv_2/biases/b_conv_2/read*/
_output_shapes
:€€€€€€€€€  *
T0
П
&conv_2/convolution_and_non_linear/ReluRelu%conv_2/convolution_and_non_linear/add*/
_output_shapes
:€€€€€€€€€  *
T0
“
conv_2/pool/MaxPoolMaxPool&conv_2/convolution_and_non_linear/Relu*
paddingSAME*
T0*
strides
*
data_formatNHWC*/
_output_shapes
:€€€€€€€€€*
ksize

~
%conv_3/weights/truncated_normal/shapeConst*%
valueB"         (   *
_output_shapes
:*
dtype0
i
$conv_3/weights/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
k
&conv_3/weights/truncated_normal/stddevConst*
valueB
 *Ќћћ=*
_output_shapes
: *
dtype0
ј
/conv_3/weights/truncated_normal/TruncatedNormalTruncatedNormal%conv_3/weights/truncated_normal/shape*

seed *
T0*
dtype0*&
_output_shapes
:(*
seed2 
і
#conv_3/weights/truncated_normal/mulMul/conv_3/weights/truncated_normal/TruncatedNormal&conv_3/weights/truncated_normal/stddev*
T0*&
_output_shapes
:(
Ґ
conv_3/weights/truncated_normalAdd#conv_3/weights/truncated_normal/mul$conv_3/weights/truncated_normal/mean*
T0*&
_output_shapes
:(
Ы
conv_3/weights/W_conv_3
VariableV2*&
_output_shapes
:(*
	container *
shape:(*
dtype0*
shared_name 
и
conv_3/weights/W_conv_3/AssignAssignconv_3/weights/W_conv_3conv_3/weights/truncated_normal**
_class 
loc:@conv_3/weights/W_conv_3*&
_output_shapes
:(*
T0*
validate_shape(*
use_locking(
Ю
conv_3/weights/W_conv_3/readIdentityconv_3/weights/W_conv_3**
_class 
loc:@conv_3/weights/W_conv_3*&
_output_shapes
:(*
T0
`
conv_3/biases/ConstConst*
valueB(*Ќћћ=*
dtype0*
_output_shapes
:(
В
conv_3/biases/b_conv_3
VariableV2*
_output_shapes
:(*
	container *
shape:(*
dtype0*
shared_name 
Ќ
conv_3/biases/b_conv_3/AssignAssignconv_3/biases/b_conv_3conv_3/biases/Const*)
_class
loc:@conv_3/biases/b_conv_3*
_output_shapes
:(*
T0*
validate_shape(*
use_locking(
П
conv_3/biases/b_conv_3/readIdentityconv_3/biases/b_conv_3*)
_class
loc:@conv_3/biases/b_conv_3*
_output_shapes
:(*
T0
ц
(conv_3/convolution_and_non_linear/Conv2DConv2Dconv_2/pool/MaxPoolconv_3/weights/W_conv_3/read*/
_output_shapes
:€€€€€€€€€(*
T0*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC*
paddingSAME
≠
%conv_3/convolution_and_non_linear/addAdd(conv_3/convolution_and_non_linear/Conv2Dconv_3/biases/b_conv_3/read*
T0*/
_output_shapes
:€€€€€€€€€(
П
&conv_3/convolution_and_non_linear/ReluRelu%conv_3/convolution_and_non_linear/add*
T0*/
_output_shapes
:€€€€€€€€€(
“
conv_3/pool/MaxPoolMaxPool&conv_3/convolution_and_non_linear/Relu*
ksize
*/
_output_shapes
:€€€€€€€€€(*
T0*
strides
*
data_formatNHWC*
paddingSAME
f
flatten/Reshape/shapeConst*
valueB"€€€€ 
  *
dtype0*
_output_shapes
:
З
flatten/ReshapeReshapeconv_3/pool/MaxPoolflatten/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
t
#fc_1/weights/truncated_normal/shapeConst*
valueB" 
  d   *
dtype0*
_output_shapes
:
g
"fc_1/weights/truncated_normal/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
i
$fc_1/weights/truncated_normal/stddevConst*
valueB
 *Ќћћ=*
_output_shapes
: *
dtype0
µ
-fc_1/weights/truncated_normal/TruncatedNormalTruncatedNormal#fc_1/weights/truncated_normal/shape*

seed *
T0*
dtype0*
_output_shapes
:	Аd*
seed2 
І
!fc_1/weights/truncated_normal/mulMul-fc_1/weights/truncated_normal/TruncatedNormal$fc_1/weights/truncated_normal/stddev*
T0*
_output_shapes
:	Аd
Х
fc_1/weights/truncated_normalAdd!fc_1/weights/truncated_normal/mul"fc_1/weights/truncated_normal/mean*
_output_shapes
:	Аd*
T0
И
fc_1/weights/W_fc1
VariableV2*
_output_shapes
:	Аd*
	container *
shape:	Аd*
dtype0*
shared_name 
–
fc_1/weights/W_fc1/AssignAssignfc_1/weights/W_fc1fc_1/weights/truncated_normal*
use_locking(*
T0*%
_class
loc:@fc_1/weights/W_fc1*
validate_shape(*
_output_shapes
:	Аd
И
fc_1/weights/W_fc1/readIdentityfc_1/weights/W_fc1*%
_class
loc:@fc_1/weights/W_fc1*
_output_shapes
:	Аd*
T0
^
fc_1/biases/ConstConst*
valueBd*Ќћћ=*
dtype0*
_output_shapes
:d
}
fc_1/biases/b_fc1
VariableV2*
shape:d*
shared_name *
dtype0*
_output_shapes
:d*
	container 
Љ
fc_1/biases/b_fc1/AssignAssignfc_1/biases/b_fc1fc_1/biases/Const*$
_class
loc:@fc_1/biases/b_fc1*
_output_shapes
:d*
T0*
validate_shape(*
use_locking(
А
fc_1/biases/b_fc1/readIdentityfc_1/biases/b_fc1*$
_class
loc:@fc_1/biases/b_fc1*
_output_shapes
:d*
T0
≠
!fc_1/matmul_and_non_linear/MatMulMatMulflatten/Reshapefc_1/weights/W_fc1/read*
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€d*
transpose_a( 
Т
fc_1/matmul_and_non_linear/addAdd!fc_1/matmul_and_non_linear/MatMulfc_1/biases/b_fc1/read*'
_output_shapes
:€€€€€€€€€d*
T0
y
fc_1/matmul_and_non_linear/ReluRelufc_1/matmul_and_non_linear/add*
T0*'
_output_shapes
:€€€€€€€€€d
t
#fc_2/weights/truncated_normal/shapeConst*
valueB"d      *
_output_shapes
:*
dtype0
g
"fc_2/weights/truncated_normal/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
i
$fc_2/weights/truncated_normal/stddevConst*
valueB
 *Ќћћ=*
_output_shapes
: *
dtype0
і
-fc_2/weights/truncated_normal/TruncatedNormalTruncatedNormal#fc_2/weights/truncated_normal/shape*
_output_shapes

:d*
seed2 *
T0*

seed *
dtype0
¶
!fc_2/weights/truncated_normal/mulMul-fc_2/weights/truncated_normal/TruncatedNormal$fc_2/weights/truncated_normal/stddev*
T0*
_output_shapes

:d
Ф
fc_2/weights/truncated_normalAdd!fc_2/weights/truncated_normal/mul"fc_2/weights/truncated_normal/mean*
_output_shapes

:d*
T0
Ж
fc_2/weights/W_fc2
VariableV2*
shape
:d*
shared_name *
dtype0*
_output_shapes

:d*
	container 
ѕ
fc_2/weights/W_fc2/AssignAssignfc_2/weights/W_fc2fc_2/weights/truncated_normal*
use_locking(*
T0*%
_class
loc:@fc_2/weights/W_fc2*
validate_shape(*
_output_shapes

:d
З
fc_2/weights/W_fc2/readIdentityfc_2/weights/W_fc2*%
_class
loc:@fc_2/weights/W_fc2*
_output_shapes

:d*
T0
^
fc_2/biases/ConstConst*
valueB*Ќћћ=*
_output_shapes
:*
dtype0
~
fc_2/biases/b_fc_2
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
њ
fc_2/biases/b_fc_2/AssignAssignfc_2/biases/b_fc_2fc_2/biases/Const*
use_locking(*
T0*%
_class
loc:@fc_2/biases/b_fc_2*
validate_shape(*
_output_shapes
:
Г
fc_2/biases/b_fc_2/readIdentityfc_2/biases/b_fc_2*
T0*%
_class
loc:@fc_2/biases/b_fc_2*
_output_shapes
:
™
outputs/MatMulMatMulfc_1/matmul_and_non_linear/Relufc_2/weights/W_fc2/read*
transpose_b( *'
_output_shapes
:€€€€€€€€€*
transpose_a( *
T0
m
outputs/addAddoutputs/MatMulfc_2/biases/b_fc_2/read*
T0*'
_output_shapes
:€€€€€€€€€
Y
outputs/SoftmaxSoftmaxoutputs/add*'
_output_shapes
:€€€€€€€€€*
T0
[
cross_entorpy/LogLogoutputs/Softmax*'
_output_shapes
:€€€€€€€€€*
T0
j
cross_entorpy/mulMulinputs/typecross_entorpy/Log*
T0*'
_output_shapes
:€€€€€€€€€
m
#cross_entorpy/Sum/reduction_indicesConst*
valueB:*
_output_shapes
:*
dtype0
Ы
cross_entorpy/SumSumcross_entorpy/mul#cross_entorpy/Sum/reduction_indices*#
_output_shapes
:€€€€€€€€€*
T0*
	keep_dims( *

Tidx0
Y
cross_entorpy/NegNegcross_entorpy/Sum*
T0*#
_output_shapes
:€€€€€€€€€
]
cross_entorpy/ConstConst*
valueB: *
_output_shapes
:*
dtype0
А
cross_entorpy/MeanMeancross_entorpy/Negcross_entorpy/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
|
 cross_entorpy/cross_entorpy/tagsConst*,
value#B! Bcross_entorpy/cross_entorpy*
dtype0*
_output_shapes
: 
Г
cross_entorpy/cross_entorpyScalarSummary cross_entorpy/cross_entorpy/tagscross_entorpy/Mean*
_output_shapes
: *
T0
X
train/gradients/ShapeConst*
valueB *
_output_shapes
: *
dtype0
Z
train/gradients/ConstConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
k
train/gradients/FillFilltrain/gradients/Shapetrain/gradients/Const*
_output_shapes
: *
T0

5train/gradients/cross_entorpy/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
Ї
/train/gradients/cross_entorpy/Mean_grad/ReshapeReshapetrain/gradients/Fill5train/gradients/cross_entorpy/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
~
-train/gradients/cross_entorpy/Mean_grad/ShapeShapecross_entorpy/Neg*
T0*
out_type0*
_output_shapes
:
‘
,train/gradients/cross_entorpy/Mean_grad/TileTile/train/gradients/cross_entorpy/Mean_grad/Reshape-train/gradients/cross_entorpy/Mean_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:€€€€€€€€€
А
/train/gradients/cross_entorpy/Mean_grad/Shape_1Shapecross_entorpy/Neg*
out_type0*
_output_shapes
:*
T0
r
/train/gradients/cross_entorpy/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
w
-train/gradients/cross_entorpy/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
“
,train/gradients/cross_entorpy/Mean_grad/ProdProd/train/gradients/cross_entorpy/Mean_grad/Shape_1-train/gradients/cross_entorpy/Mean_grad/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
y
/train/gradients/cross_entorpy/Mean_grad/Const_1Const*
valueB: *
_output_shapes
:*
dtype0
÷
.train/gradients/cross_entorpy/Mean_grad/Prod_1Prod/train/gradients/cross_entorpy/Mean_grad/Shape_2/train/gradients/cross_entorpy/Mean_grad/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
s
1train/gradients/cross_entorpy/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
Њ
/train/gradients/cross_entorpy/Mean_grad/MaximumMaximum.train/gradients/cross_entorpy/Mean_grad/Prod_11train/gradients/cross_entorpy/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
Љ
0train/gradients/cross_entorpy/Mean_grad/floordivFloorDiv,train/gradients/cross_entorpy/Mean_grad/Prod/train/gradients/cross_entorpy/Mean_grad/Maximum*
T0*
_output_shapes
: 
Ц
,train/gradients/cross_entorpy/Mean_grad/CastCast0train/gradients/cross_entorpy/Mean_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0
ƒ
/train/gradients/cross_entorpy/Mean_grad/truedivRealDiv,train/gradients/cross_entorpy/Mean_grad/Tile,train/gradients/cross_entorpy/Mean_grad/Cast*#
_output_shapes
:€€€€€€€€€*
T0
Р
*train/gradients/cross_entorpy/Neg_grad/NegNeg/train/gradients/cross_entorpy/Mean_grad/truediv*#
_output_shapes
:€€€€€€€€€*
T0
}
,train/gradients/cross_entorpy/Sum_grad/ShapeShapecross_entorpy/mul*
out_type0*
_output_shapes
:*
T0
m
+train/gradients/cross_entorpy/Sum_grad/SizeConst*
value	B :*
_output_shapes
: *
dtype0
®
*train/gradients/cross_entorpy/Sum_grad/addAdd#cross_entorpy/Sum/reduction_indices+train/gradients/cross_entorpy/Sum_grad/Size*
T0*
_output_shapes
:
і
*train/gradients/cross_entorpy/Sum_grad/modFloorMod*train/gradients/cross_entorpy/Sum_grad/add+train/gradients/cross_entorpy/Sum_grad/Size*
_output_shapes
:*
T0
x
.train/gradients/cross_entorpy/Sum_grad/Shape_1Const*
valueB:*
_output_shapes
:*
dtype0
t
2train/gradients/cross_entorpy/Sum_grad/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
t
2train/gradients/cross_entorpy/Sum_grad/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
т
,train/gradients/cross_entorpy/Sum_grad/rangeRange2train/gradients/cross_entorpy/Sum_grad/range/start+train/gradients/cross_entorpy/Sum_grad/Size2train/gradients/cross_entorpy/Sum_grad/range/delta*
_output_shapes
:*

Tidx0
s
1train/gradients/cross_entorpy/Sum_grad/Fill/valueConst*
value	B :*
dtype0*
_output_shapes
: 
ї
+train/gradients/cross_entorpy/Sum_grad/FillFill.train/gradients/cross_entorpy/Sum_grad/Shape_11train/gradients/cross_entorpy/Sum_grad/Fill/value*
T0*
_output_shapes
:
±
4train/gradients/cross_entorpy/Sum_grad/DynamicStitchDynamicStitch,train/gradients/cross_entorpy/Sum_grad/range*train/gradients/cross_entorpy/Sum_grad/mod,train/gradients/cross_entorpy/Sum_grad/Shape+train/gradients/cross_entorpy/Sum_grad/Fill*#
_output_shapes
:€€€€€€€€€*
T0*
N
r
0train/gradients/cross_entorpy/Sum_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
ѕ
.train/gradients/cross_entorpy/Sum_grad/MaximumMaximum4train/gradients/cross_entorpy/Sum_grad/DynamicStitch0train/gradients/cross_entorpy/Sum_grad/Maximum/y*
T0*#
_output_shapes
:€€€€€€€€€
Њ
/train/gradients/cross_entorpy/Sum_grad/floordivFloorDiv,train/gradients/cross_entorpy/Sum_grad/Shape.train/gradients/cross_entorpy/Sum_grad/Maximum*
_output_shapes
:*
T0
ћ
.train/gradients/cross_entorpy/Sum_grad/ReshapeReshape*train/gradients/cross_entorpy/Neg_grad/Neg4train/gradients/cross_entorpy/Sum_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
Ў
+train/gradients/cross_entorpy/Sum_grad/TileTile.train/gradients/cross_entorpy/Sum_grad/Reshape/train/gradients/cross_entorpy/Sum_grad/floordiv*'
_output_shapes
:€€€€€€€€€*
T0*

Tmultiples0
w
,train/gradients/cross_entorpy/mul_grad/ShapeShapeinputs/type*
out_type0*
_output_shapes
:*
T0

.train/gradients/cross_entorpy/mul_grad/Shape_1Shapecross_entorpy/Log*
T0*
out_type0*
_output_shapes
:
р
<train/gradients/cross_entorpy/mul_grad/BroadcastGradientArgsBroadcastGradientArgs,train/gradients/cross_entorpy/mul_grad/Shape.train/gradients/cross_entorpy/mul_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
£
*train/gradients/cross_entorpy/mul_grad/mulMul+train/gradients/cross_entorpy/Sum_grad/Tilecross_entorpy/Log*'
_output_shapes
:€€€€€€€€€*
T0
џ
*train/gradients/cross_entorpy/mul_grad/SumSum*train/gradients/cross_entorpy/mul_grad/mul<train/gradients/cross_entorpy/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
”
.train/gradients/cross_entorpy/mul_grad/ReshapeReshape*train/gradients/cross_entorpy/mul_grad/Sum,train/gradients/cross_entorpy/mul_grad/Shape*
Tshape0*'
_output_shapes
:€€€€€€€€€*
T0
Я
,train/gradients/cross_entorpy/mul_grad/mul_1Mulinputs/type+train/gradients/cross_entorpy/Sum_grad/Tile*'
_output_shapes
:€€€€€€€€€*
T0
б
,train/gradients/cross_entorpy/mul_grad/Sum_1Sum,train/gradients/cross_entorpy/mul_grad/mul_1>train/gradients/cross_entorpy/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
ў
0train/gradients/cross_entorpy/mul_grad/Reshape_1Reshape,train/gradients/cross_entorpy/mul_grad/Sum_1.train/gradients/cross_entorpy/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
£
7train/gradients/cross_entorpy/mul_grad/tuple/group_depsNoOp/^train/gradients/cross_entorpy/mul_grad/Reshape1^train/gradients/cross_entorpy/mul_grad/Reshape_1
™
?train/gradients/cross_entorpy/mul_grad/tuple/control_dependencyIdentity.train/gradients/cross_entorpy/mul_grad/Reshape8^train/gradients/cross_entorpy/mul_grad/tuple/group_deps*A
_class7
53loc:@train/gradients/cross_entorpy/mul_grad/Reshape*'
_output_shapes
:€€€€€€€€€*
T0
∞
Atrain/gradients/cross_entorpy/mul_grad/tuple/control_dependency_1Identity0train/gradients/cross_entorpy/mul_grad/Reshape_18^train/gradients/cross_entorpy/mul_grad/tuple/group_deps*C
_class9
75loc:@train/gradients/cross_entorpy/mul_grad/Reshape_1*'
_output_shapes
:€€€€€€€€€*
T0
∆
1train/gradients/cross_entorpy/Log_grad/Reciprocal
Reciprocaloutputs/SoftmaxB^train/gradients/cross_entorpy/mul_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:€€€€€€€€€
ў
*train/gradients/cross_entorpy/Log_grad/mulMulAtrain/gradients/cross_entorpy/mul_grad/tuple/control_dependency_11train/gradients/cross_entorpy/Log_grad/Reciprocal*
T0*'
_output_shapes
:€€€€€€€€€
Ю
(train/gradients/outputs/Softmax_grad/mulMul*train/gradients/cross_entorpy/Log_grad/muloutputs/Softmax*'
_output_shapes
:€€€€€€€€€*
T0
Д
:train/gradients/outputs/Softmax_grad/Sum/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
а
(train/gradients/outputs/Softmax_grad/SumSum(train/gradients/outputs/Softmax_grad/mul:train/gradients/outputs/Softmax_grad/Sum/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:€€€€€€€€€
Г
2train/gradients/outputs/Softmax_grad/Reshape/shapeConst*
valueB"€€€€   *
_output_shapes
:*
dtype0
’
,train/gradients/outputs/Softmax_grad/ReshapeReshape(train/gradients/outputs/Softmax_grad/Sum2train/gradients/outputs/Softmax_grad/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
ї
(train/gradients/outputs/Softmax_grad/subSub*train/gradients/cross_entorpy/Log_grad/mul,train/gradients/outputs/Softmax_grad/Reshape*
T0*'
_output_shapes
:€€€€€€€€€
Ю
*train/gradients/outputs/Softmax_grad/mul_1Mul(train/gradients/outputs/Softmax_grad/suboutputs/Softmax*'
_output_shapes
:€€€€€€€€€*
T0
t
&train/gradients/outputs/add_grad/ShapeShapeoutputs/MatMul*
out_type0*
_output_shapes
:*
T0
r
(train/gradients/outputs/add_grad/Shape_1Const*
valueB:*
_output_shapes
:*
dtype0
ё
6train/gradients/outputs/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/outputs/add_grad/Shape(train/gradients/outputs/add_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
ѕ
$train/gradients/outputs/add_grad/SumSum*train/gradients/outputs/Softmax_grad/mul_16train/gradients/outputs/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ѕ
(train/gradients/outputs/add_grad/ReshapeReshape$train/gradients/outputs/add_grad/Sum&train/gradients/outputs/add_grad/Shape*
Tshape0*'
_output_shapes
:€€€€€€€€€*
T0
”
&train/gradients/outputs/add_grad/Sum_1Sum*train/gradients/outputs/Softmax_grad/mul_18train/gradients/outputs/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ї
*train/gradients/outputs/add_grad/Reshape_1Reshape&train/gradients/outputs/add_grad/Sum_1(train/gradients/outputs/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
С
1train/gradients/outputs/add_grad/tuple/group_depsNoOp)^train/gradients/outputs/add_grad/Reshape+^train/gradients/outputs/add_grad/Reshape_1
Т
9train/gradients/outputs/add_grad/tuple/control_dependencyIdentity(train/gradients/outputs/add_grad/Reshape2^train/gradients/outputs/add_grad/tuple/group_deps*;
_class1
/-loc:@train/gradients/outputs/add_grad/Reshape*'
_output_shapes
:€€€€€€€€€*
T0
Л
;train/gradients/outputs/add_grad/tuple/control_dependency_1Identity*train/gradients/outputs/add_grad/Reshape_12^train/gradients/outputs/add_grad/tuple/group_deps*=
_class3
1/loc:@train/gradients/outputs/add_grad/Reshape_1*
_output_shapes
:*
T0
а
*train/gradients/outputs/MatMul_grad/MatMulMatMul9train/gradients/outputs/add_grad/tuple/control_dependencyfc_2/weights/W_fc2/read*
transpose_b(*
T0*'
_output_shapes
:€€€€€€€€€d*
transpose_a( 
б
,train/gradients/outputs/MatMul_grad/MatMul_1MatMulfc_1/matmul_and_non_linear/Relu9train/gradients/outputs/add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:d*
transpose_a(
Ш
4train/gradients/outputs/MatMul_grad/tuple/group_depsNoOp+^train/gradients/outputs/MatMul_grad/MatMul-^train/gradients/outputs/MatMul_grad/MatMul_1
Ь
<train/gradients/outputs/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/outputs/MatMul_grad/MatMul5^train/gradients/outputs/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/outputs/MatMul_grad/MatMul*'
_output_shapes
:€€€€€€€€€d
Щ
>train/gradients/outputs/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/outputs/MatMul_grad/MatMul_15^train/gradients/outputs/MatMul_grad/tuple/group_deps*?
_class5
31loc:@train/gradients/outputs/MatMul_grad/MatMul_1*
_output_shapes

:d*
T0
Џ
=train/gradients/fc_1/matmul_and_non_linear/Relu_grad/ReluGradReluGrad<train/gradients/outputs/MatMul_grad/tuple/control_dependencyfc_1/matmul_and_non_linear/Relu*'
_output_shapes
:€€€€€€€€€d*
T0
Ъ
9train/gradients/fc_1/matmul_and_non_linear/add_grad/ShapeShape!fc_1/matmul_and_non_linear/MatMul*
out_type0*
_output_shapes
:*
T0
Е
;train/gradients/fc_1/matmul_and_non_linear/add_grad/Shape_1Const*
valueB:d*
_output_shapes
:*
dtype0
Ч
Itrain/gradients/fc_1/matmul_and_non_linear/add_grad/BroadcastGradientArgsBroadcastGradientArgs9train/gradients/fc_1/matmul_and_non_linear/add_grad/Shape;train/gradients/fc_1/matmul_and_non_linear/add_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
И
7train/gradients/fc_1/matmul_and_non_linear/add_grad/SumSum=train/gradients/fc_1/matmul_and_non_linear/Relu_grad/ReluGradItrain/gradients/fc_1/matmul_and_non_linear/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ъ
;train/gradients/fc_1/matmul_and_non_linear/add_grad/ReshapeReshape7train/gradients/fc_1/matmul_and_non_linear/add_grad/Sum9train/gradients/fc_1/matmul_and_non_linear/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€d
М
9train/gradients/fc_1/matmul_and_non_linear/add_grad/Sum_1Sum=train/gradients/fc_1/matmul_and_non_linear/Relu_grad/ReluGradKtrain/gradients/fc_1/matmul_and_non_linear/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
у
=train/gradients/fc_1/matmul_and_non_linear/add_grad/Reshape_1Reshape9train/gradients/fc_1/matmul_and_non_linear/add_grad/Sum_1;train/gradients/fc_1/matmul_and_non_linear/add_grad/Shape_1*
Tshape0*
_output_shapes
:d*
T0
 
Dtrain/gradients/fc_1/matmul_and_non_linear/add_grad/tuple/group_depsNoOp<^train/gradients/fc_1/matmul_and_non_linear/add_grad/Reshape>^train/gradients/fc_1/matmul_and_non_linear/add_grad/Reshape_1
ё
Ltrain/gradients/fc_1/matmul_and_non_linear/add_grad/tuple/control_dependencyIdentity;train/gradients/fc_1/matmul_and_non_linear/add_grad/ReshapeE^train/gradients/fc_1/matmul_and_non_linear/add_grad/tuple/group_deps*
T0*N
_classD
B@loc:@train/gradients/fc_1/matmul_and_non_linear/add_grad/Reshape*'
_output_shapes
:€€€€€€€€€d
„
Ntrain/gradients/fc_1/matmul_and_non_linear/add_grad/tuple/control_dependency_1Identity=train/gradients/fc_1/matmul_and_non_linear/add_grad/Reshape_1E^train/gradients/fc_1/matmul_and_non_linear/add_grad/tuple/group_deps*P
_classF
DBloc:@train/gradients/fc_1/matmul_and_non_linear/add_grad/Reshape_1*
_output_shapes
:d*
T0
З
=train/gradients/fc_1/matmul_and_non_linear/MatMul_grad/MatMulMatMulLtrain/gradients/fc_1/matmul_and_non_linear/add_grad/tuple/control_dependencyfc_1/weights/W_fc1/read*
transpose_b(*(
_output_shapes
:€€€€€€€€€А*
transpose_a( *
T0
ш
?train/gradients/fc_1/matmul_and_non_linear/MatMul_grad/MatMul_1MatMulflatten/ReshapeLtrain/gradients/fc_1/matmul_and_non_linear/add_grad/tuple/control_dependency*
transpose_b( *
_output_shapes
:	Аd*
transpose_a(*
T0
—
Gtrain/gradients/fc_1/matmul_and_non_linear/MatMul_grad/tuple/group_depsNoOp>^train/gradients/fc_1/matmul_and_non_linear/MatMul_grad/MatMul@^train/gradients/fc_1/matmul_and_non_linear/MatMul_grad/MatMul_1
й
Otrain/gradients/fc_1/matmul_and_non_linear/MatMul_grad/tuple/control_dependencyIdentity=train/gradients/fc_1/matmul_and_non_linear/MatMul_grad/MatMulH^train/gradients/fc_1/matmul_and_non_linear/MatMul_grad/tuple/group_deps*
T0*P
_classF
DBloc:@train/gradients/fc_1/matmul_and_non_linear/MatMul_grad/MatMul*(
_output_shapes
:€€€€€€€€€А
ж
Qtrain/gradients/fc_1/matmul_and_non_linear/MatMul_grad/tuple/control_dependency_1Identity?train/gradients/fc_1/matmul_and_non_linear/MatMul_grad/MatMul_1H^train/gradients/fc_1/matmul_and_non_linear/MatMul_grad/tuple/group_deps*R
_classH
FDloc:@train/gradients/fc_1/matmul_and_non_linear/MatMul_grad/MatMul_1*
_output_shapes
:	Аd*
T0
}
*train/gradients/flatten/Reshape_grad/ShapeShapeconv_3/pool/MaxPool*
T0*
out_type0*
_output_shapes
:
ь
,train/gradients/flatten/Reshape_grad/ReshapeReshapeOtrain/gradients/fc_1/matmul_and_non_linear/MatMul_grad/tuple/control_dependency*train/gradients/flatten/Reshape_grad/Shape*
Tshape0*/
_output_shapes
:€€€€€€€€€(*
T0
Ї
4train/gradients/conv_3/pool/MaxPool_grad/MaxPoolGradMaxPoolGrad&conv_3/convolution_and_non_linear/Reluconv_3/pool/MaxPool,train/gradients/flatten/Reshape_grad/Reshape*
ksize
*/
_output_shapes
:€€€€€€€€€(*
T0*
data_formatNHWC*
strides
*
paddingSAME
и
Dtrain/gradients/conv_3/convolution_and_non_linear/Relu_grad/ReluGradReluGrad4train/gradients/conv_3/pool/MaxPool_grad/MaxPoolGrad&conv_3/convolution_and_non_linear/Relu*
T0*/
_output_shapes
:€€€€€€€€€(
®
@train/gradients/conv_3/convolution_and_non_linear/add_grad/ShapeShape(conv_3/convolution_and_non_linear/Conv2D*
out_type0*
_output_shapes
:*
T0
М
Btrain/gradients/conv_3/convolution_and_non_linear/add_grad/Shape_1Const*
valueB:(*
dtype0*
_output_shapes
:
ђ
Ptrain/gradients/conv_3/convolution_and_non_linear/add_grad/BroadcastGradientArgsBroadcastGradientArgs@train/gradients/conv_3/convolution_and_non_linear/add_grad/ShapeBtrain/gradients/conv_3/convolution_and_non_linear/add_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Э
>train/gradients/conv_3/convolution_and_non_linear/add_grad/SumSumDtrain/gradients/conv_3/convolution_and_non_linear/Relu_grad/ReluGradPtrain/gradients/conv_3/convolution_and_non_linear/add_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ч
Btrain/gradients/conv_3/convolution_and_non_linear/add_grad/ReshapeReshape>train/gradients/conv_3/convolution_and_non_linear/add_grad/Sum@train/gradients/conv_3/convolution_and_non_linear/add_grad/Shape*
T0*
Tshape0*/
_output_shapes
:€€€€€€€€€(
°
@train/gradients/conv_3/convolution_and_non_linear/add_grad/Sum_1SumDtrain/gradients/conv_3/convolution_and_non_linear/Relu_grad/ReluGradRtrain/gradients/conv_3/convolution_and_non_linear/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
И
Dtrain/gradients/conv_3/convolution_and_non_linear/add_grad/Reshape_1Reshape@train/gradients/conv_3/convolution_and_non_linear/add_grad/Sum_1Btrain/gradients/conv_3/convolution_and_non_linear/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:(
я
Ktrain/gradients/conv_3/convolution_and_non_linear/add_grad/tuple/group_depsNoOpC^train/gradients/conv_3/convolution_and_non_linear/add_grad/ReshapeE^train/gradients/conv_3/convolution_and_non_linear/add_grad/Reshape_1
В
Strain/gradients/conv_3/convolution_and_non_linear/add_grad/tuple/control_dependencyIdentityBtrain/gradients/conv_3/convolution_and_non_linear/add_grad/ReshapeL^train/gradients/conv_3/convolution_and_non_linear/add_grad/tuple/group_deps*
T0*U
_classK
IGloc:@train/gradients/conv_3/convolution_and_non_linear/add_grad/Reshape*/
_output_shapes
:€€€€€€€€€(
у
Utrain/gradients/conv_3/convolution_and_non_linear/add_grad/tuple/control_dependency_1IdentityDtrain/gradients/conv_3/convolution_and_non_linear/add_grad/Reshape_1L^train/gradients/conv_3/convolution_and_non_linear/add_grad/tuple/group_deps*W
_classM
KIloc:@train/gradients/conv_3/convolution_and_non_linear/add_grad/Reshape_1*
_output_shapes
:(*
T0
Ц
Ctrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/ShapeShapeconv_2/pool/MaxPool*
T0*
out_type0*
_output_shapes
:
ћ
Qtrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputCtrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/Shapeconv_3/weights/W_conv_3/readStrain/gradients/conv_3/convolution_and_non_linear/add_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
T0*
paddingSAME*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
data_formatNHWC*
strides

Ю
Etrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/Shape_1Const*%
valueB"         (   *
_output_shapes
:*
dtype0
£
Rtrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterconv_2/pool/MaxPoolEtrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/Shape_1Strain/gradients/conv_3/convolution_and_non_linear/add_grad/tuple/control_dependency*
paddingSAME*
T0*
data_formatNHWC*
strides
*&
_output_shapes
:(*
use_cudnn_on_gpu(
€
Ntrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/tuple/group_depsNoOpR^train/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInputS^train/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilter
¶
Vtrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/tuple/control_dependencyIdentityQtrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInputO^train/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/tuple/group_deps*d
_classZ
XVloc:@train/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:€€€€€€€€€*
T0
°
Xtrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/tuple/control_dependency_1IdentityRtrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilterO^train/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/tuple/group_deps*
T0*e
_class[
YWloc:@train/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:(
д
4train/gradients/conv_2/pool/MaxPool_grad/MaxPoolGradMaxPoolGrad&conv_2/convolution_and_non_linear/Reluconv_2/pool/MaxPoolVtrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:€€€€€€€€€  
и
Dtrain/gradients/conv_2/convolution_and_non_linear/Relu_grad/ReluGradReluGrad4train/gradients/conv_2/pool/MaxPool_grad/MaxPoolGrad&conv_2/convolution_and_non_linear/Relu*
T0*/
_output_shapes
:€€€€€€€€€  
®
@train/gradients/conv_2/convolution_and_non_linear/add_grad/ShapeShape(conv_2/convolution_and_non_linear/Conv2D*
out_type0*
_output_shapes
:*
T0
М
Btrain/gradients/conv_2/convolution_and_non_linear/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
ђ
Ptrain/gradients/conv_2/convolution_and_non_linear/add_grad/BroadcastGradientArgsBroadcastGradientArgs@train/gradients/conv_2/convolution_and_non_linear/add_grad/ShapeBtrain/gradients/conv_2/convolution_and_non_linear/add_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Э
>train/gradients/conv_2/convolution_and_non_linear/add_grad/SumSumDtrain/gradients/conv_2/convolution_and_non_linear/Relu_grad/ReluGradPtrain/gradients/conv_2/convolution_and_non_linear/add_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ч
Btrain/gradients/conv_2/convolution_and_non_linear/add_grad/ReshapeReshape>train/gradients/conv_2/convolution_and_non_linear/add_grad/Sum@train/gradients/conv_2/convolution_and_non_linear/add_grad/Shape*
T0*
Tshape0*/
_output_shapes
:€€€€€€€€€  
°
@train/gradients/conv_2/convolution_and_non_linear/add_grad/Sum_1SumDtrain/gradients/conv_2/convolution_and_non_linear/Relu_grad/ReluGradRtrain/gradients/conv_2/convolution_and_non_linear/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
И
Dtrain/gradients/conv_2/convolution_and_non_linear/add_grad/Reshape_1Reshape@train/gradients/conv_2/convolution_and_non_linear/add_grad/Sum_1Btrain/gradients/conv_2/convolution_and_non_linear/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
я
Ktrain/gradients/conv_2/convolution_and_non_linear/add_grad/tuple/group_depsNoOpC^train/gradients/conv_2/convolution_and_non_linear/add_grad/ReshapeE^train/gradients/conv_2/convolution_and_non_linear/add_grad/Reshape_1
В
Strain/gradients/conv_2/convolution_and_non_linear/add_grad/tuple/control_dependencyIdentityBtrain/gradients/conv_2/convolution_and_non_linear/add_grad/ReshapeL^train/gradients/conv_2/convolution_and_non_linear/add_grad/tuple/group_deps*U
_classK
IGloc:@train/gradients/conv_2/convolution_and_non_linear/add_grad/Reshape*/
_output_shapes
:€€€€€€€€€  *
T0
у
Utrain/gradients/conv_2/convolution_and_non_linear/add_grad/tuple/control_dependency_1IdentityDtrain/gradients/conv_2/convolution_and_non_linear/add_grad/Reshape_1L^train/gradients/conv_2/convolution_and_non_linear/add_grad/tuple/group_deps*
T0*W
_classM
KIloc:@train/gradients/conv_2/convolution_and_non_linear/add_grad/Reshape_1*
_output_shapes
:
Ц
Ctrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/ShapeShapeconv_1/pool/MaxPool*
T0*
out_type0*
_output_shapes
:
ћ
Qtrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputCtrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/Shapeconv_2/weights/W_conv_2/readStrain/gradients/conv_2/convolution_and_non_linear/add_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
T0*
paddingSAME*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
data_formatNHWC*
strides

Ю
Etrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/Shape_1Const*%
valueB"      
      *
dtype0*
_output_shapes
:
£
Rtrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterconv_1/pool/MaxPoolEtrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/Shape_1Strain/gradients/conv_2/convolution_and_non_linear/add_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
T0*
paddingSAME*&
_output_shapes
:
*
data_formatNHWC*
strides

€
Ntrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/tuple/group_depsNoOpR^train/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInputS^train/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilter
¶
Vtrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/tuple/control_dependencyIdentityQtrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInputO^train/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@train/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:€€€€€€€€€  

°
Xtrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/tuple/control_dependency_1IdentityRtrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilterO^train/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/tuple/group_deps*e
_class[
YWloc:@train/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:
*
T0
ж
4train/gradients/conv_1/pool/MaxPool_grad/MaxPoolGradMaxPoolGrad&conv_1/convolution_and_non_linear/Reluconv_1/pool/MaxPoolVtrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/tuple/control_dependency*
ksize
*1
_output_shapes
:€€€€€€€€€АА
*
T0*
data_formatNHWC*
strides
*
paddingSAME
к
Dtrain/gradients/conv_1/convolution_and_non_linear/Relu_grad/ReluGradReluGrad4train/gradients/conv_1/pool/MaxPool_grad/MaxPoolGrad&conv_1/convolution_and_non_linear/Relu*
T0*1
_output_shapes
:€€€€€€€€€АА

®
@train/gradients/conv_1/convolution_and_non_linear/add_grad/ShapeShape(conv_1/convolution_and_non_linear/Conv2D*
out_type0*
_output_shapes
:*
T0
М
Btrain/gradients/conv_1/convolution_and_non_linear/add_grad/Shape_1Const*
valueB:
*
_output_shapes
:*
dtype0
ђ
Ptrain/gradients/conv_1/convolution_and_non_linear/add_grad/BroadcastGradientArgsBroadcastGradientArgs@train/gradients/conv_1/convolution_and_non_linear/add_grad/ShapeBtrain/gradients/conv_1/convolution_and_non_linear/add_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
Э
>train/gradients/conv_1/convolution_and_non_linear/add_grad/SumSumDtrain/gradients/conv_1/convolution_and_non_linear/Relu_grad/ReluGradPtrain/gradients/conv_1/convolution_and_non_linear/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Щ
Btrain/gradients/conv_1/convolution_and_non_linear/add_grad/ReshapeReshape>train/gradients/conv_1/convolution_and_non_linear/add_grad/Sum@train/gradients/conv_1/convolution_and_non_linear/add_grad/Shape*
Tshape0*1
_output_shapes
:€€€€€€€€€АА
*
T0
°
@train/gradients/conv_1/convolution_and_non_linear/add_grad/Sum_1SumDtrain/gradients/conv_1/convolution_and_non_linear/Relu_grad/ReluGradRtrain/gradients/conv_1/convolution_and_non_linear/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
И
Dtrain/gradients/conv_1/convolution_and_non_linear/add_grad/Reshape_1Reshape@train/gradients/conv_1/convolution_and_non_linear/add_grad/Sum_1Btrain/gradients/conv_1/convolution_and_non_linear/add_grad/Shape_1*
Tshape0*
_output_shapes
:
*
T0
я
Ktrain/gradients/conv_1/convolution_and_non_linear/add_grad/tuple/group_depsNoOpC^train/gradients/conv_1/convolution_and_non_linear/add_grad/ReshapeE^train/gradients/conv_1/convolution_and_non_linear/add_grad/Reshape_1
Д
Strain/gradients/conv_1/convolution_and_non_linear/add_grad/tuple/control_dependencyIdentityBtrain/gradients/conv_1/convolution_and_non_linear/add_grad/ReshapeL^train/gradients/conv_1/convolution_and_non_linear/add_grad/tuple/group_deps*U
_classK
IGloc:@train/gradients/conv_1/convolution_and_non_linear/add_grad/Reshape*1
_output_shapes
:€€€€€€€€€АА
*
T0
у
Utrain/gradients/conv_1/convolution_and_non_linear/add_grad/tuple/control_dependency_1IdentityDtrain/gradients/conv_1/convolution_and_non_linear/add_grad/Reshape_1L^train/gradients/conv_1/convolution_and_non_linear/add_grad/tuple/group_deps*W
_classM
KIloc:@train/gradients/conv_1/convolution_and_non_linear/add_grad/Reshape_1*
_output_shapes
:
*
T0
Ъ
Ctrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/ShapeShapeprepare_tensors/Reshape*
out_type0*
_output_shapes
:*
T0
ћ
Qtrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputCtrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/Shapeconv_1/weights/W_conv_1/readStrain/gradients/conv_1/convolution_and_non_linear/add_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ю
Etrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/Shape_1Const*%
valueB"         
   *
dtype0*
_output_shapes
:
І
Rtrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterprepare_tensors/ReshapeEtrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/Shape_1Strain/gradients/conv_1/convolution_and_non_linear/add_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:

€
Ntrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/tuple/group_depsNoOpR^train/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInputS^train/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilter
®
Vtrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/tuple/control_dependencyIdentityQtrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInputO^train/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@train/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInput*1
_output_shapes
:€€€€€€€€€АА
°
Xtrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/tuple/control_dependency_1IdentityRtrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilterO^train/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/tuple/group_deps*
T0*e
_class[
YWloc:@train/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:

Р
train/beta1_power/initial_valueConst*
valueB
 *fff?**
_class 
loc:@conv_1/weights/W_conv_1*
dtype0*
_output_shapes
: 
°
train/beta1_power
VariableV2*
shape: *
_output_shapes
: *
shared_name **
_class 
loc:@conv_1/weights/W_conv_1*
dtype0*
	container 
ћ
train/beta1_power/AssignAssigntrain/beta1_powertrain/beta1_power/initial_value**
_class 
loc:@conv_1/weights/W_conv_1*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
В
train/beta1_power/readIdentitytrain/beta1_power*
T0**
_class 
loc:@conv_1/weights/W_conv_1*
_output_shapes
: 
Р
train/beta2_power/initial_valueConst*
valueB
 *wЊ?**
_class 
loc:@conv_1/weights/W_conv_1*
dtype0*
_output_shapes
: 
°
train/beta2_power
VariableV2*
shape: *
_output_shapes
: *
shared_name **
_class 
loc:@conv_1/weights/W_conv_1*
dtype0*
	container 
ћ
train/beta2_power/AssignAssigntrain/beta2_powertrain/beta2_power/initial_value**
_class 
loc:@conv_1/weights/W_conv_1*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
В
train/beta2_power/readIdentitytrain/beta2_power**
_class 
loc:@conv_1/weights/W_conv_1*
_output_shapes
: *
T0
p
train/zerosConst*%
valueB
*    *
dtype0*&
_output_shapes
:

ћ
conv_1/weights/W_conv_1/Adam
VariableV2*
shared_name **
_class 
loc:@conv_1/weights/W_conv_1*
	container *
shape:
*
dtype0*&
_output_shapes
:

ё
#conv_1/weights/W_conv_1/Adam/AssignAssignconv_1/weights/W_conv_1/Adamtrain/zeros*
use_locking(*
T0**
_class 
loc:@conv_1/weights/W_conv_1*
validate_shape(*&
_output_shapes
:

®
!conv_1/weights/W_conv_1/Adam/readIdentityconv_1/weights/W_conv_1/Adam**
_class 
loc:@conv_1/weights/W_conv_1*&
_output_shapes
:
*
T0
r
train/zeros_1Const*%
valueB
*    *&
_output_shapes
:
*
dtype0
ќ
conv_1/weights/W_conv_1/Adam_1
VariableV2*
	container *
dtype0**
_class 
loc:@conv_1/weights/W_conv_1*&
_output_shapes
:
*
shape:
*
shared_name 
д
%conv_1/weights/W_conv_1/Adam_1/AssignAssignconv_1/weights/W_conv_1/Adam_1train/zeros_1*
use_locking(*
T0**
_class 
loc:@conv_1/weights/W_conv_1*
validate_shape(*&
_output_shapes
:

ђ
#conv_1/weights/W_conv_1/Adam_1/readIdentityconv_1/weights/W_conv_1/Adam_1**
_class 
loc:@conv_1/weights/W_conv_1*&
_output_shapes
:
*
T0
Z
train/zeros_2Const*
valueB
*    *
_output_shapes
:
*
dtype0
≤
conv_1/biases/b_conv_1/Adam
VariableV2*
	container *
dtype0*)
_class
loc:@conv_1/biases/b_conv_1*
_output_shapes
:
*
shape:
*
shared_name 
—
"conv_1/biases/b_conv_1/Adam/AssignAssignconv_1/biases/b_conv_1/Adamtrain/zeros_2*
use_locking(*
T0*)
_class
loc:@conv_1/biases/b_conv_1*
validate_shape(*
_output_shapes
:

Щ
 conv_1/biases/b_conv_1/Adam/readIdentityconv_1/biases/b_conv_1/Adam*
T0*)
_class
loc:@conv_1/biases/b_conv_1*
_output_shapes
:

Z
train/zeros_3Const*
valueB
*    *
dtype0*
_output_shapes
:

і
conv_1/biases/b_conv_1/Adam_1
VariableV2*)
_class
loc:@conv_1/biases/b_conv_1*
_output_shapes
:
*
shape:
*
dtype0*
shared_name *
	container 
’
$conv_1/biases/b_conv_1/Adam_1/AssignAssignconv_1/biases/b_conv_1/Adam_1train/zeros_3*
use_locking(*
T0*)
_class
loc:@conv_1/biases/b_conv_1*
validate_shape(*
_output_shapes
:

Э
"conv_1/biases/b_conv_1/Adam_1/readIdentityconv_1/biases/b_conv_1/Adam_1*)
_class
loc:@conv_1/biases/b_conv_1*
_output_shapes
:
*
T0
r
train/zeros_4Const*%
valueB
*    *
dtype0*&
_output_shapes
:

ћ
conv_2/weights/W_conv_2/Adam
VariableV2*
shared_name **
_class 
loc:@conv_2/weights/W_conv_2*
	container *
shape:
*
dtype0*&
_output_shapes
:

а
#conv_2/weights/W_conv_2/Adam/AssignAssignconv_2/weights/W_conv_2/Adamtrain/zeros_4*
use_locking(*
T0**
_class 
loc:@conv_2/weights/W_conv_2*
validate_shape(*&
_output_shapes
:

®
!conv_2/weights/W_conv_2/Adam/readIdentityconv_2/weights/W_conv_2/Adam*
T0**
_class 
loc:@conv_2/weights/W_conv_2*&
_output_shapes
:

r
train/zeros_5Const*%
valueB
*    *&
_output_shapes
:
*
dtype0
ќ
conv_2/weights/W_conv_2/Adam_1
VariableV2**
_class 
loc:@conv_2/weights/W_conv_2*&
_output_shapes
:
*
shape:
*
dtype0*
shared_name *
	container 
д
%conv_2/weights/W_conv_2/Adam_1/AssignAssignconv_2/weights/W_conv_2/Adam_1train/zeros_5*
use_locking(*
T0**
_class 
loc:@conv_2/weights/W_conv_2*
validate_shape(*&
_output_shapes
:

ђ
#conv_2/weights/W_conv_2/Adam_1/readIdentityconv_2/weights/W_conv_2/Adam_1**
_class 
loc:@conv_2/weights/W_conv_2*&
_output_shapes
:
*
T0
Z
train/zeros_6Const*
valueB*    *
_output_shapes
:*
dtype0
≤
conv_2/biases/b_conv_2/Adam
VariableV2*)
_class
loc:@conv_2/biases/b_conv_2*
_output_shapes
:*
shape:*
dtype0*
shared_name *
	container 
—
"conv_2/biases/b_conv_2/Adam/AssignAssignconv_2/biases/b_conv_2/Adamtrain/zeros_6*)
_class
loc:@conv_2/biases/b_conv_2*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
Щ
 conv_2/biases/b_conv_2/Adam/readIdentityconv_2/biases/b_conv_2/Adam*
T0*)
_class
loc:@conv_2/biases/b_conv_2*
_output_shapes
:
Z
train/zeros_7Const*
valueB*    *
_output_shapes
:*
dtype0
і
conv_2/biases/b_conv_2/Adam_1
VariableV2*
	container *
dtype0*)
_class
loc:@conv_2/biases/b_conv_2*
_output_shapes
:*
shape:*
shared_name 
’
$conv_2/biases/b_conv_2/Adam_1/AssignAssignconv_2/biases/b_conv_2/Adam_1train/zeros_7*)
_class
loc:@conv_2/biases/b_conv_2*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
Э
"conv_2/biases/b_conv_2/Adam_1/readIdentityconv_2/biases/b_conv_2/Adam_1*)
_class
loc:@conv_2/biases/b_conv_2*
_output_shapes
:*
T0
r
train/zeros_8Const*%
valueB(*    *
dtype0*&
_output_shapes
:(
ћ
conv_3/weights/W_conv_3/Adam
VariableV2*
shape:(*&
_output_shapes
:(*
shared_name **
_class 
loc:@conv_3/weights/W_conv_3*
dtype0*
	container 
а
#conv_3/weights/W_conv_3/Adam/AssignAssignconv_3/weights/W_conv_3/Adamtrain/zeros_8**
_class 
loc:@conv_3/weights/W_conv_3*&
_output_shapes
:(*
T0*
validate_shape(*
use_locking(
®
!conv_3/weights/W_conv_3/Adam/readIdentityconv_3/weights/W_conv_3/Adam*
T0**
_class 
loc:@conv_3/weights/W_conv_3*&
_output_shapes
:(
r
train/zeros_9Const*%
valueB(*    *
dtype0*&
_output_shapes
:(
ќ
conv_3/weights/W_conv_3/Adam_1
VariableV2*
	container *
dtype0**
_class 
loc:@conv_3/weights/W_conv_3*&
_output_shapes
:(*
shape:(*
shared_name 
д
%conv_3/weights/W_conv_3/Adam_1/AssignAssignconv_3/weights/W_conv_3/Adam_1train/zeros_9**
_class 
loc:@conv_3/weights/W_conv_3*&
_output_shapes
:(*
T0*
validate_shape(*
use_locking(
ђ
#conv_3/weights/W_conv_3/Adam_1/readIdentityconv_3/weights/W_conv_3/Adam_1*
T0**
_class 
loc:@conv_3/weights/W_conv_3*&
_output_shapes
:(
[
train/zeros_10Const*
valueB(*    *
_output_shapes
:(*
dtype0
≤
conv_3/biases/b_conv_3/Adam
VariableV2*
	container *
dtype0*)
_class
loc:@conv_3/biases/b_conv_3*
_output_shapes
:(*
shape:(*
shared_name 
“
"conv_3/biases/b_conv_3/Adam/AssignAssignconv_3/biases/b_conv_3/Adamtrain/zeros_10*
use_locking(*
T0*)
_class
loc:@conv_3/biases/b_conv_3*
validate_shape(*
_output_shapes
:(
Щ
 conv_3/biases/b_conv_3/Adam/readIdentityconv_3/biases/b_conv_3/Adam*
T0*)
_class
loc:@conv_3/biases/b_conv_3*
_output_shapes
:(
[
train/zeros_11Const*
valueB(*    *
_output_shapes
:(*
dtype0
і
conv_3/biases/b_conv_3/Adam_1
VariableV2*
shape:(*
_output_shapes
:(*
shared_name *)
_class
loc:@conv_3/biases/b_conv_3*
dtype0*
	container 
÷
$conv_3/biases/b_conv_3/Adam_1/AssignAssignconv_3/biases/b_conv_3/Adam_1train/zeros_11*)
_class
loc:@conv_3/biases/b_conv_3*
_output_shapes
:(*
T0*
validate_shape(*
use_locking(
Э
"conv_3/biases/b_conv_3/Adam_1/readIdentityconv_3/biases/b_conv_3/Adam_1*)
_class
loc:@conv_3/biases/b_conv_3*
_output_shapes
:(*
T0
e
train/zeros_12Const*
valueB	Аd*    *
dtype0*
_output_shapes
:	Аd
і
fc_1/weights/W_fc1/Adam
VariableV2*%
_class
loc:@fc_1/weights/W_fc1*
_output_shapes
:	Аd*
shape:	Аd*
dtype0*
shared_name *
	container 
Ћ
fc_1/weights/W_fc1/Adam/AssignAssignfc_1/weights/W_fc1/Adamtrain/zeros_12*%
_class
loc:@fc_1/weights/W_fc1*
_output_shapes
:	Аd*
T0*
validate_shape(*
use_locking(
Т
fc_1/weights/W_fc1/Adam/readIdentityfc_1/weights/W_fc1/Adam*
T0*%
_class
loc:@fc_1/weights/W_fc1*
_output_shapes
:	Аd
e
train/zeros_13Const*
valueB	Аd*    *
_output_shapes
:	Аd*
dtype0
ґ
fc_1/weights/W_fc1/Adam_1
VariableV2*
shared_name *%
_class
loc:@fc_1/weights/W_fc1*
	container *
shape:	Аd*
dtype0*
_output_shapes
:	Аd
ѕ
 fc_1/weights/W_fc1/Adam_1/AssignAssignfc_1/weights/W_fc1/Adam_1train/zeros_13*
use_locking(*
T0*%
_class
loc:@fc_1/weights/W_fc1*
validate_shape(*
_output_shapes
:	Аd
Ц
fc_1/weights/W_fc1/Adam_1/readIdentityfc_1/weights/W_fc1/Adam_1*
T0*%
_class
loc:@fc_1/weights/W_fc1*
_output_shapes
:	Аd
[
train/zeros_14Const*
valueBd*    *
dtype0*
_output_shapes
:d
®
fc_1/biases/b_fc1/Adam
VariableV2*
	container *
dtype0*$
_class
loc:@fc_1/biases/b_fc1*
_output_shapes
:d*
shape:d*
shared_name 
√
fc_1/biases/b_fc1/Adam/AssignAssignfc_1/biases/b_fc1/Adamtrain/zeros_14*$
_class
loc:@fc_1/biases/b_fc1*
_output_shapes
:d*
T0*
validate_shape(*
use_locking(
К
fc_1/biases/b_fc1/Adam/readIdentityfc_1/biases/b_fc1/Adam*$
_class
loc:@fc_1/biases/b_fc1*
_output_shapes
:d*
T0
[
train/zeros_15Const*
valueBd*    *
_output_shapes
:d*
dtype0
™
fc_1/biases/b_fc1/Adam_1
VariableV2*
	container *
dtype0*$
_class
loc:@fc_1/biases/b_fc1*
_output_shapes
:d*
shape:d*
shared_name 
«
fc_1/biases/b_fc1/Adam_1/AssignAssignfc_1/biases/b_fc1/Adam_1train/zeros_15*$
_class
loc:@fc_1/biases/b_fc1*
_output_shapes
:d*
T0*
validate_shape(*
use_locking(
О
fc_1/biases/b_fc1/Adam_1/readIdentityfc_1/biases/b_fc1/Adam_1*$
_class
loc:@fc_1/biases/b_fc1*
_output_shapes
:d*
T0
c
train/zeros_16Const*
valueBd*    *
_output_shapes

:d*
dtype0
≤
fc_2/weights/W_fc2/Adam
VariableV2*
shared_name *%
_class
loc:@fc_2/weights/W_fc2*
	container *
shape
:d*
dtype0*
_output_shapes

:d
 
fc_2/weights/W_fc2/Adam/AssignAssignfc_2/weights/W_fc2/Adamtrain/zeros_16*%
_class
loc:@fc_2/weights/W_fc2*
_output_shapes

:d*
T0*
validate_shape(*
use_locking(
С
fc_2/weights/W_fc2/Adam/readIdentityfc_2/weights/W_fc2/Adam*%
_class
loc:@fc_2/weights/W_fc2*
_output_shapes

:d*
T0
c
train/zeros_17Const*
valueBd*    *
_output_shapes

:d*
dtype0
і
fc_2/weights/W_fc2/Adam_1
VariableV2*
shape
:d*
_output_shapes

:d*
shared_name *%
_class
loc:@fc_2/weights/W_fc2*
dtype0*
	container 
ќ
 fc_2/weights/W_fc2/Adam_1/AssignAssignfc_2/weights/W_fc2/Adam_1train/zeros_17*
use_locking(*
T0*%
_class
loc:@fc_2/weights/W_fc2*
validate_shape(*
_output_shapes

:d
Х
fc_2/weights/W_fc2/Adam_1/readIdentityfc_2/weights/W_fc2/Adam_1*
T0*%
_class
loc:@fc_2/weights/W_fc2*
_output_shapes

:d
[
train/zeros_18Const*
valueB*    *
_output_shapes
:*
dtype0
™
fc_2/biases/b_fc_2/Adam
VariableV2*
shape:*
_output_shapes
:*
shared_name *%
_class
loc:@fc_2/biases/b_fc_2*
dtype0*
	container 
∆
fc_2/biases/b_fc_2/Adam/AssignAssignfc_2/biases/b_fc_2/Adamtrain/zeros_18*
use_locking(*
T0*%
_class
loc:@fc_2/biases/b_fc_2*
validate_shape(*
_output_shapes
:
Н
fc_2/biases/b_fc_2/Adam/readIdentityfc_2/biases/b_fc_2/Adam*
T0*%
_class
loc:@fc_2/biases/b_fc_2*
_output_shapes
:
[
train/zeros_19Const*
valueB*    *
dtype0*
_output_shapes
:
ђ
fc_2/biases/b_fc_2/Adam_1
VariableV2*
	container *
dtype0*%
_class
loc:@fc_2/biases/b_fc_2*
_output_shapes
:*
shape:*
shared_name 
 
 fc_2/biases/b_fc_2/Adam_1/AssignAssignfc_2/biases/b_fc_2/Adam_1train/zeros_19*
use_locking(*
T0*%
_class
loc:@fc_2/biases/b_fc_2*
validate_shape(*
_output_shapes
:
С
fc_2/biases/b_fc_2/Adam_1/readIdentityfc_2/biases/b_fc_2/Adam_1*%
_class
loc:@fc_2/biases/b_fc_2*
_output_shapes
:*
T0
]
train/Adam/learning_rateConst*
valueB
 *Ј—8*
_output_shapes
: *
dtype0
U
train/Adam/beta1Const*
valueB
 *fff?*
_output_shapes
: *
dtype0
U
train/Adam/beta2Const*
valueB
 *wЊ?*
dtype0*
_output_shapes
: 
W
train/Adam/epsilonConst*
valueB
 *wћ+2*
_output_shapes
: *
dtype0
г
3train/Adam/update_conv_1/weights/W_conv_1/ApplyAdam	ApplyAdamconv_1/weights/W_conv_1conv_1/weights/W_conv_1/Adamconv_1/weights/W_conv_1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonXtrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/tuple/control_dependency_1**
_class 
loc:@conv_1/weights/W_conv_1*&
_output_shapes
:
*
T0*
use_locking( 
ѕ
2train/Adam/update_conv_1/biases/b_conv_1/ApplyAdam	ApplyAdamconv_1/biases/b_conv_1conv_1/biases/b_conv_1/Adamconv_1/biases/b_conv_1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonUtrain/gradients/conv_1/convolution_and_non_linear/add_grad/tuple/control_dependency_1*)
_class
loc:@conv_1/biases/b_conv_1*
_output_shapes
:
*
T0*
use_locking( 
г
3train/Adam/update_conv_2/weights/W_conv_2/ApplyAdam	ApplyAdamconv_2/weights/W_conv_2conv_2/weights/W_conv_2/Adamconv_2/weights/W_conv_2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonXtrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/tuple/control_dependency_1**
_class 
loc:@conv_2/weights/W_conv_2*&
_output_shapes
:
*
T0*
use_locking( 
ѕ
2train/Adam/update_conv_2/biases/b_conv_2/ApplyAdam	ApplyAdamconv_2/biases/b_conv_2conv_2/biases/b_conv_2/Adamconv_2/biases/b_conv_2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonUtrain/gradients/conv_2/convolution_and_non_linear/add_grad/tuple/control_dependency_1*)
_class
loc:@conv_2/biases/b_conv_2*
_output_shapes
:*
T0*
use_locking( 
г
3train/Adam/update_conv_3/weights/W_conv_3/ApplyAdam	ApplyAdamconv_3/weights/W_conv_3conv_3/weights/W_conv_3/Adamconv_3/weights/W_conv_3/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonXtrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/tuple/control_dependency_1**
_class 
loc:@conv_3/weights/W_conv_3*&
_output_shapes
:(*
T0*
use_locking( 
ѕ
2train/Adam/update_conv_3/biases/b_conv_3/ApplyAdam	ApplyAdamconv_3/biases/b_conv_3conv_3/biases/b_conv_3/Adamconv_3/biases/b_conv_3/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonUtrain/gradients/conv_3/convolution_and_non_linear/add_grad/tuple/control_dependency_1*)
_class
loc:@conv_3/biases/b_conv_3*
_output_shapes
:(*
T0*
use_locking( 
Љ
.train/Adam/update_fc_1/weights/W_fc1/ApplyAdam	ApplyAdamfc_1/weights/W_fc1fc_1/weights/W_fc1/Adamfc_1/weights/W_fc1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonQtrain/gradients/fc_1/matmul_and_non_linear/MatMul_grad/tuple/control_dependency_1*%
_class
loc:@fc_1/weights/W_fc1*
_output_shapes
:	Аd*
T0*
use_locking( 
ѓ
-train/Adam/update_fc_1/biases/b_fc1/ApplyAdam	ApplyAdamfc_1/biases/b_fc1fc_1/biases/b_fc1/Adamfc_1/biases/b_fc1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonNtrain/gradients/fc_1/matmul_and_non_linear/add_grad/tuple/control_dependency_1*$
_class
loc:@fc_1/biases/b_fc1*
_output_shapes
:d*
T0*
use_locking( 
®
.train/Adam/update_fc_2/weights/W_fc2/ApplyAdam	ApplyAdamfc_2/weights/W_fc2fc_2/weights/W_fc2/Adamfc_2/weights/W_fc2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/outputs/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*%
_class
loc:@fc_2/weights/W_fc2*
_output_shapes

:d
°
.train/Adam/update_fc_2/biases/b_fc_2/ApplyAdam	ApplyAdamfc_2/biases/b_fc_2fc_2/biases/b_fc_2/Adamfc_2/biases/b_fc_2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/outputs/add_grad/tuple/control_dependency_1*
use_locking( *
T0*%
_class
loc:@fc_2/biases/b_fc_2*
_output_shapes
:
Р
train/Adam/mulMultrain/beta1_power/readtrain/Adam/beta14^train/Adam/update_conv_1/weights/W_conv_1/ApplyAdam3^train/Adam/update_conv_1/biases/b_conv_1/ApplyAdam4^train/Adam/update_conv_2/weights/W_conv_2/ApplyAdam3^train/Adam/update_conv_2/biases/b_conv_2/ApplyAdam4^train/Adam/update_conv_3/weights/W_conv_3/ApplyAdam3^train/Adam/update_conv_3/biases/b_conv_3/ApplyAdam/^train/Adam/update_fc_1/weights/W_fc1/ApplyAdam.^train/Adam/update_fc_1/biases/b_fc1/ApplyAdam/^train/Adam/update_fc_2/weights/W_fc2/ApplyAdam/^train/Adam/update_fc_2/biases/b_fc_2/ApplyAdam**
_class 
loc:@conv_1/weights/W_conv_1*
_output_shapes
: *
T0
і
train/Adam/AssignAssigntrain/beta1_powertrain/Adam/mul*
use_locking( *
T0**
_class 
loc:@conv_1/weights/W_conv_1*
validate_shape(*
_output_shapes
: 
Т
train/Adam/mul_1Multrain/beta2_power/readtrain/Adam/beta24^train/Adam/update_conv_1/weights/W_conv_1/ApplyAdam3^train/Adam/update_conv_1/biases/b_conv_1/ApplyAdam4^train/Adam/update_conv_2/weights/W_conv_2/ApplyAdam3^train/Adam/update_conv_2/biases/b_conv_2/ApplyAdam4^train/Adam/update_conv_3/weights/W_conv_3/ApplyAdam3^train/Adam/update_conv_3/biases/b_conv_3/ApplyAdam/^train/Adam/update_fc_1/weights/W_fc1/ApplyAdam.^train/Adam/update_fc_1/biases/b_fc1/ApplyAdam/^train/Adam/update_fc_2/weights/W_fc2/ApplyAdam/^train/Adam/update_fc_2/biases/b_fc_2/ApplyAdam*
T0**
_class 
loc:@conv_1/weights/W_conv_1*
_output_shapes
: 
Є
train/Adam/Assign_1Assigntrain/beta2_powertrain/Adam/mul_1**
_class 
loc:@conv_1/weights/W_conv_1*
_output_shapes
: *
T0*
validate_shape(*
use_locking( 
ј

train/AdamNoOp4^train/Adam/update_conv_1/weights/W_conv_1/ApplyAdam3^train/Adam/update_conv_1/biases/b_conv_1/ApplyAdam4^train/Adam/update_conv_2/weights/W_conv_2/ApplyAdam3^train/Adam/update_conv_2/biases/b_conv_2/ApplyAdam4^train/Adam/update_conv_3/weights/W_conv_3/ApplyAdam3^train/Adam/update_conv_3/biases/b_conv_3/ApplyAdam/^train/Adam/update_fc_1/weights/W_fc1/ApplyAdam.^train/Adam/update_fc_1/biases/b_fc1/ApplyAdam/^train/Adam/update_fc_2/weights/W_fc2/ApplyAdam/^train/Adam/update_fc_2/biases/b_fc_2/ApplyAdam^train/Adam/Assign^train/Adam/Assign_1
[
evaluate/ArgMax/dimensionConst*
value	B :*
_output_shapes
: *
dtype0

evaluate/ArgMaxArgMaxoutputs/Softmaxevaluate/ArgMax/dimension*#
_output_shapes
:€€€€€€€€€*
T0*

Tidx0
]
evaluate/ArgMax_1/dimensionConst*
value	B :*
_output_shapes
: *
dtype0

evaluate/ArgMax_1ArgMaxinputs/typeevaluate/ArgMax_1/dimension*

Tidx0*
T0*#
_output_shapes
:€€€€€€€€€
i
evaluate/EqualEqualevaluate/ArgMaxevaluate/ArgMax_1*#
_output_shapes
:€€€€€€€€€*
T0	
b
evaluate/CastCastevaluate/Equal*

SrcT0
*#
_output_shapes
:€€€€€€€€€*

DstT0
X
evaluate/ConstConst*
valueB: *
dtype0*
_output_shapes
:
r
evaluate/MeanMeanevaluate/Castevaluate/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
`
Merge/MergeSummaryMergeSummarycross_entorpy/cross_entorpy*
_output_shapes
: *
N"".
	summaries!

cross_entorpy/cross_entorpy:0"х
trainable_variablesЁЏ
[
conv_1/weights/W_conv_1:0conv_1/weights/W_conv_1/Assignconv_1/weights/W_conv_1/read:0
X
conv_1/biases/b_conv_1:0conv_1/biases/b_conv_1/Assignconv_1/biases/b_conv_1/read:0
[
conv_2/weights/W_conv_2:0conv_2/weights/W_conv_2/Assignconv_2/weights/W_conv_2/read:0
X
conv_2/biases/b_conv_2:0conv_2/biases/b_conv_2/Assignconv_2/biases/b_conv_2/read:0
[
conv_3/weights/W_conv_3:0conv_3/weights/W_conv_3/Assignconv_3/weights/W_conv_3/read:0
X
conv_3/biases/b_conv_3:0conv_3/biases/b_conv_3/Assignconv_3/biases/b_conv_3/read:0
L
fc_1/weights/W_fc1:0fc_1/weights/W_fc1/Assignfc_1/weights/W_fc1/read:0
I
fc_1/biases/b_fc1:0fc_1/biases/b_fc1/Assignfc_1/biases/b_fc1/read:0
L
fc_2/weights/W_fc2:0fc_2/weights/W_fc2/Assignfc_2/weights/W_fc2/read:0
L
fc_2/biases/b_fc_2:0fc_2/biases/b_fc_2/Assignfc_2/biases/b_fc_2/read:0"
train_op


train/Adam"Э
	variablesПМ
[
conv_1/weights/W_conv_1:0conv_1/weights/W_conv_1/Assignconv_1/weights/W_conv_1/read:0
X
conv_1/biases/b_conv_1:0conv_1/biases/b_conv_1/Assignconv_1/biases/b_conv_1/read:0
[
conv_2/weights/W_conv_2:0conv_2/weights/W_conv_2/Assignconv_2/weights/W_conv_2/read:0
X
conv_2/biases/b_conv_2:0conv_2/biases/b_conv_2/Assignconv_2/biases/b_conv_2/read:0
[
conv_3/weights/W_conv_3:0conv_3/weights/W_conv_3/Assignconv_3/weights/W_conv_3/read:0
X
conv_3/biases/b_conv_3:0conv_3/biases/b_conv_3/Assignconv_3/biases/b_conv_3/read:0
L
fc_1/weights/W_fc1:0fc_1/weights/W_fc1/Assignfc_1/weights/W_fc1/read:0
I
fc_1/biases/b_fc1:0fc_1/biases/b_fc1/Assignfc_1/biases/b_fc1/read:0
L
fc_2/weights/W_fc2:0fc_2/weights/W_fc2/Assignfc_2/weights/W_fc2/read:0
L
fc_2/biases/b_fc_2:0fc_2/biases/b_fc_2/Assignfc_2/biases/b_fc_2/read:0
I
train/beta1_power:0train/beta1_power/Assigntrain/beta1_power/read:0
I
train/beta2_power:0train/beta2_power/Assigntrain/beta2_power/read:0
j
conv_1/weights/W_conv_1/Adam:0#conv_1/weights/W_conv_1/Adam/Assign#conv_1/weights/W_conv_1/Adam/read:0
p
 conv_1/weights/W_conv_1/Adam_1:0%conv_1/weights/W_conv_1/Adam_1/Assign%conv_1/weights/W_conv_1/Adam_1/read:0
g
conv_1/biases/b_conv_1/Adam:0"conv_1/biases/b_conv_1/Adam/Assign"conv_1/biases/b_conv_1/Adam/read:0
m
conv_1/biases/b_conv_1/Adam_1:0$conv_1/biases/b_conv_1/Adam_1/Assign$conv_1/biases/b_conv_1/Adam_1/read:0
j
conv_2/weights/W_conv_2/Adam:0#conv_2/weights/W_conv_2/Adam/Assign#conv_2/weights/W_conv_2/Adam/read:0
p
 conv_2/weights/W_conv_2/Adam_1:0%conv_2/weights/W_conv_2/Adam_1/Assign%conv_2/weights/W_conv_2/Adam_1/read:0
g
conv_2/biases/b_conv_2/Adam:0"conv_2/biases/b_conv_2/Adam/Assign"conv_2/biases/b_conv_2/Adam/read:0
m
conv_2/biases/b_conv_2/Adam_1:0$conv_2/biases/b_conv_2/Adam_1/Assign$conv_2/biases/b_conv_2/Adam_1/read:0
j
conv_3/weights/W_conv_3/Adam:0#conv_3/weights/W_conv_3/Adam/Assign#conv_3/weights/W_conv_3/Adam/read:0
p
 conv_3/weights/W_conv_3/Adam_1:0%conv_3/weights/W_conv_3/Adam_1/Assign%conv_3/weights/W_conv_3/Adam_1/read:0
g
conv_3/biases/b_conv_3/Adam:0"conv_3/biases/b_conv_3/Adam/Assign"conv_3/biases/b_conv_3/Adam/read:0
m
conv_3/biases/b_conv_3/Adam_1:0$conv_3/biases/b_conv_3/Adam_1/Assign$conv_3/biases/b_conv_3/Adam_1/read:0
[
fc_1/weights/W_fc1/Adam:0fc_1/weights/W_fc1/Adam/Assignfc_1/weights/W_fc1/Adam/read:0
a
fc_1/weights/W_fc1/Adam_1:0 fc_1/weights/W_fc1/Adam_1/Assign fc_1/weights/W_fc1/Adam_1/read:0
X
fc_1/biases/b_fc1/Adam:0fc_1/biases/b_fc1/Adam/Assignfc_1/biases/b_fc1/Adam/read:0
^
fc_1/biases/b_fc1/Adam_1:0fc_1/biases/b_fc1/Adam_1/Assignfc_1/biases/b_fc1/Adam_1/read:0
[
fc_2/weights/W_fc2/Adam:0fc_2/weights/W_fc2/Adam/Assignfc_2/weights/W_fc2/Adam/read:0
a
fc_2/weights/W_fc2/Adam_1:0 fc_2/weights/W_fc2/Adam_1/Assign fc_2/weights/W_fc2/Adam_1/read:0
[
fc_2/biases/b_fc_2/Adam:0fc_2/biases/b_fc_2/Adam/Assignfc_2/biases/b_fc_2/Adam/read:0
a
fc_2/biases/b_fc_2/Adam_1:0 fc_2/biases/b_fc_2/Adam_1/Assign fc_2/biases/b_fc_2/Adam_1/read:0∆Q§&/       m]P	≠2°£•D÷A*$
"
cross_entorpy/cross_entorpyж `?O∆^1       ЭГГй	”љƒ•D÷A*$
"
cross_entorpy/cross_entorpy$?&•1       ЭГГй	RGЬд•D÷A(*$
"
cross_entorpy/cross_entorpyй_?8й≥1       ЭГГй	б¶D÷A<*$
"
cross_entorpy/cross_entorpyM?]Yљ1       ЭГГй	m;©%¶D÷AP*$
"
cross_entorpy/cross_entorpyЄ-о>аф1       ЭГГй	:WF¶D÷Ad*$
"
cross_entorpy/cross_entorpyЬ)й>ЄI1       ЭГГй	Eї≥f¶D÷Ax*$
"
cross_entorpy/cross_entorpy§{“>Ы@∞Э2       $Vм	3э2З¶D÷AМ*$
"
cross_entorpy/cross_entorpy№у≈>©32       $Vм	©2≠І¶D÷A†*$
"
cross_entorpy/cross_entorpy°eЅ>q7ф+2       $Vм	Б"E»¶D÷Aі*$
"
cross_entorpy/cross_entorpy¶≥∞> Q0З2       $Vм	гэ и¶D÷A»*$
"
cross_entorpy/cross_entorpyЌI™>Ђ
 <2       $Vм	мC	ІD÷A№*$
"
cross_entorpy/cross_entorpyП"Ц>£
Dx2       $Vм	РЌ)ІD÷Aр*$
"
cross_entorpy/cross_entorpyюgК>≠вь2       $Vм	±NJІD÷AД*$
"
cross_entorpy/cross_entorpyџЅЖ>∆х%2       $Vм	вw≈jІD÷AШ*$
"
cross_entorpy/cross_entorpyПФO>ЧќщВ2       $Vм	KиUЛІD÷Aђ*$
"
cross_entorpy/cross_entorpy¶BD>A«цh2       $Vм	№БжЂІD÷Aј*$
"
cross_entorpy/cross_entorpyмZ>U0C2       $Vм	XюsћІD÷A‘*$
"
cross_entorpy/cross_entorpyx√'>я№Ѕ2       $Vм	t(ьмІD÷Aи*$
"
cross_entorpy/cross_entorpydµ>ю£з2       $Vм	6rz®D÷Aь*$
"
cross_entorpy/cross_entorpyйяр==n6c2       $Vм	`з.®D÷AР*$
"
cross_entorpy/cross_entorpyCЃ	>w∞2       $Vм	Ѓ-ОN®D÷A§*$
"
cross_entorpy/cross_entorpy±><g"j2       $Vм	^o®D÷AЄ*$
"
cross_entorpy/cross_entorpyЌ€–=NЈсђ2       $Vм	÷УП®D÷Aћ*$
"
cross_entorpy/cross_entorpyqѓЏ=hdXn2       $Vм	e5∞®D÷Aа*$
"
cross_entorpy/cross_entorpyqаШ=cГ’o2       $Vм	}йЭ–®D÷Aф*$
"
cross_entorpy/cross_entorpy≈ўЪ=єБx–2       $Vм	уѓ"с®D÷AИ*$
"
cross_entorpy/cross_entorpy»kЧ=@Ід2       $Vм	8jФ©D÷AЬ*$
"
cross_entorpy/cross_entorpyбh=Мzя2       $Vм	су2©D÷A∞*$
"
cross_entorpy/cross_entorpyНhX=э√“П2       $Vм	$‘ХR©D÷Aƒ*$
"
cross_entorpy/cross_entorpyt%-=	Еiђ2       $Vм	цFs©D÷AЎ*$
"
cross_entorpy/cross_entorpyбК'=xћF2       $Vм	4ШУ©D÷Aм*$
"
cross_entorpy/cross_entorpy~%=Kl!2       $Vм	…+і©D÷AА*$
"
cross_entorpy/cross_entorpydv=j3;з2       $Vм	√Ы¶‘©D÷AФ*$
"
cross_entorpy/cross_entorpyі≠=iк2       $Vм	yУ,х©D÷A®*$
"
cross_entorpy/cross_entorpyцµп<©„VЕ2       $Vм	ф£І™D÷AЉ*$
"
cross_entorpy/cross_entorpyr¬№<жЩѓw2       $Vм	.»"6™D÷A–*$
"
cross_entorpy/cross_entorpyТZЌ<’ХO2       $Vм	хЕЂV™D÷Aд*$
"
cross_entorpy/cross_entorpyZƒє<Zэ„2       $Vм	њоw™D÷Aш*$
"
cross_entorpy/cross_entorpyьЫ<ƒ“/з2       $Vм	ЉћЪЧ™D÷AМ*$
"
cross_entorpy/cross_entorpyУ≠Т<ђ“@2       $Vм	~ЛЄ™D÷A†*$
"
cross_entorpy/cross_entorpy\пq<НZEc2       $Vм	hY™Ў™D÷Aі*$
"
cross_entorpy/cross_entorpyY<И:ВЛ2       $Vм	ƒB-щ™D÷A»*$
"
cross_entorpy/cross_entorpyџa<НZ(ь2       $Vм	0ЉЇЂD÷A№*$
"
cross_entorpy/cross_entorpy!O<џ†џ2       $Vм	Б"K:ЂD÷Aр*$
"
cross_entorpy/cross_entorpyд*F<sYу2       $Vм	≈v«ZЂD÷AД*$
"
cross_entorpy/cross_entorpy Ђ-<шS2       $Vм	ЏYB{ЂD÷AШ*$
"
cross_entorpy/cross_entorpy$Э<Pƒ4»2       $Vм	IЮљЫЂD÷Aђ*$
"
cross_entorpy/cross_entorpyЅ…<mр…u2       $Vм	4iJЉЂD÷Aј*$
"
cross_entorpy/cross_entorpye/ч;R{ЅI2       $Vм	ТСќ№ЂD÷A‘*$
"
cross_entorpy/cross_entorpy3Пщ;:@хШ2       $Vм	4’BэЂD÷Aи*$
"
cross_entorpy/cross_entorpy Ђа;яPzо