       ЃK"	  РфБBжAbrain.Event:2ћ[ъ      ЖћЙ§	6гфБBжA"Б
a
inputs/graphsPlaceholder*)
_output_shapes
:џџџџџџџџџ*
dtype0*
shape: 
]
inputs/typePlaceholder*
shape: *
dtype0*'
_output_shapes
:џџџџџџџџџ
v
prepare_tensors/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"џџџџ         

prepare_tensors/ReshapeReshapeinputs/graphsprepare_tensors/Reshape/shape*
T0*
Tshape0*1
_output_shapes
:џџџџџџџџџ
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
&conv_1/weights/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *ЭЬЬ=
Р
/conv_1/weights/truncated_normal/TruncatedNormalTruncatedNormal%conv_1/weights/truncated_normal/shape*

seed *
T0*
dtype0*&
_output_shapes
:
*
seed2 
Д
#conv_1/weights/truncated_normal/mulMul/conv_1/weights/truncated_normal/TruncatedNormal&conv_1/weights/truncated_normal/stddev*
T0*&
_output_shapes
:

Ђ
conv_1/weights/truncated_normalAdd#conv_1/weights/truncated_normal/mul$conv_1/weights/truncated_normal/mean*
T0*&
_output_shapes
:


conv_1/weights/W_conv_1
VariableV2*&
_output_shapes
:
*
	container *
shape:
*
dtype0*
shared_name 
ш
conv_1/weights/W_conv_1/AssignAssignconv_1/weights/W_conv_1conv_1/weights/truncated_normal*
use_locking(*
T0**
_class 
loc:@conv_1/weights/W_conv_1*
validate_shape(*&
_output_shapes
:


conv_1/weights/W_conv_1/readIdentityconv_1/weights/W_conv_1*&
_output_shapes
:
**
_class 
loc:@conv_1/weights/W_conv_1*
T0
`
conv_1/biases/ConstConst*
valueB
*ЭЬЬ=*
_output_shapes
:
*
dtype0

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
Э
conv_1/biases/b_conv_1/AssignAssignconv_1/biases/b_conv_1conv_1/biases/Const*
_output_shapes
:
*
validate_shape(*)
_class
loc:@conv_1/biases/b_conv_1*
T0*
use_locking(

conv_1/biases/b_conv_1/readIdentityconv_1/biases/b_conv_1*
T0*)
_class
loc:@conv_1/biases/b_conv_1*
_output_shapes
:

ќ
(conv_1/convolution_and_non_linear/Conv2DConv2Dprepare_tensors/Reshapeconv_1/weights/W_conv_1/read*
strides
*
data_formatNHWC*1
_output_shapes
:џџџџџџџџџ
*
paddingSAME*
T0*
use_cudnn_on_gpu(
Џ
%conv_1/convolution_and_non_linear/addAdd(conv_1/convolution_and_non_linear/Conv2Dconv_1/biases/b_conv_1/read*
T0*1
_output_shapes
:џџџџџџџџџ


&conv_1/convolution_and_non_linear/ReluRelu%conv_1/convolution_and_non_linear/add*1
_output_shapes
:џџџџџџџџџ
*
T0
в
conv_1/pool/MaxPoolMaxPool&conv_1/convolution_and_non_linear/Relu*
ksize
*/
_output_shapes
:џџџџџџџџџ  
*
strides
*
data_formatNHWC*
T0*
paddingSAME
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
 *ЭЬЬ=*
_output_shapes
: *
dtype0
Р
/conv_2/weights/truncated_normal/TruncatedNormalTruncatedNormal%conv_2/weights/truncated_normal/shape*

seed *
T0*
dtype0*&
_output_shapes
:
*
seed2 
Д
#conv_2/weights/truncated_normal/mulMul/conv_2/weights/truncated_normal/TruncatedNormal&conv_2/weights/truncated_normal/stddev*&
_output_shapes
:
*
T0
Ђ
conv_2/weights/truncated_normalAdd#conv_2/weights/truncated_normal/mul$conv_2/weights/truncated_normal/mean*&
_output_shapes
:
*
T0

conv_2/weights/W_conv_2
VariableV2*
shape:
*
shared_name *
dtype0*&
_output_shapes
:
*
	container 
ш
conv_2/weights/W_conv_2/AssignAssignconv_2/weights/W_conv_2conv_2/weights/truncated_normal**
_class 
loc:@conv_2/weights/W_conv_2*&
_output_shapes
:
*
T0*
validate_shape(*
use_locking(

conv_2/weights/W_conv_2/readIdentityconv_2/weights/W_conv_2**
_class 
loc:@conv_2/weights/W_conv_2*&
_output_shapes
:
*
T0
`
conv_2/biases/ConstConst*
dtype0*
_output_shapes
:*
valueB*ЭЬЬ=

conv_2/biases/b_conv_2
VariableV2*
_output_shapes
:*
	container *
shape:*
dtype0*
shared_name 
Э
conv_2/biases/b_conv_2/AssignAssignconv_2/biases/b_conv_2conv_2/biases/Const*)
_class
loc:@conv_2/biases/b_conv_2*
_output_shapes
:*
T0*
validate_shape(*
use_locking(

conv_2/biases/b_conv_2/readIdentityconv_2/biases/b_conv_2*
T0*)
_class
loc:@conv_2/biases/b_conv_2*
_output_shapes
:
і
(conv_2/convolution_and_non_linear/Conv2DConv2Dconv_1/pool/MaxPoolconv_2/weights/W_conv_2/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:џџџџџџџџџ  
­
%conv_2/convolution_and_non_linear/addAdd(conv_2/convolution_and_non_linear/Conv2Dconv_2/biases/b_conv_2/read*
T0*/
_output_shapes
:џџџџџџџџџ  

&conv_2/convolution_and_non_linear/ReluRelu%conv_2/convolution_and_non_linear/add*/
_output_shapes
:џџџџџџџџџ  *
T0
в
conv_2/pool/MaxPoolMaxPool&conv_2/convolution_and_non_linear/Relu*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
ksize
*
strides
*
data_formatNHWC*
T0
~
%conv_3/weights/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*%
valueB"         (   
i
$conv_3/weights/truncated_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
k
&conv_3/weights/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *ЭЬЬ=
Р
/conv_3/weights/truncated_normal/TruncatedNormalTruncatedNormal%conv_3/weights/truncated_normal/shape*

seed *
T0*
dtype0*&
_output_shapes
:(*
seed2 
Д
#conv_3/weights/truncated_normal/mulMul/conv_3/weights/truncated_normal/TruncatedNormal&conv_3/weights/truncated_normal/stddev*
T0*&
_output_shapes
:(
Ђ
conv_3/weights/truncated_normalAdd#conv_3/weights/truncated_normal/mul$conv_3/weights/truncated_normal/mean*&
_output_shapes
:(*
T0

conv_3/weights/W_conv_3
VariableV2*&
_output_shapes
:(*
	container *
shape:(*
dtype0*
shared_name 
ш
conv_3/weights/W_conv_3/AssignAssignconv_3/weights/W_conv_3conv_3/weights/truncated_normal*
use_locking(*
validate_shape(*
T0*&
_output_shapes
:(**
_class 
loc:@conv_3/weights/W_conv_3

conv_3/weights/W_conv_3/readIdentityconv_3/weights/W_conv_3*
T0*&
_output_shapes
:(**
_class 
loc:@conv_3/weights/W_conv_3
`
conv_3/biases/ConstConst*
dtype0*
_output_shapes
:(*
valueB(*ЭЬЬ=

conv_3/biases/b_conv_3
VariableV2*
_output_shapes
:(*
	container *
shape:(*
dtype0*
shared_name 
Э
conv_3/biases/b_conv_3/AssignAssignconv_3/biases/b_conv_3conv_3/biases/Const*
_output_shapes
:(*
validate_shape(*)
_class
loc:@conv_3/biases/b_conv_3*
T0*
use_locking(

conv_3/biases/b_conv_3/readIdentityconv_3/biases/b_conv_3*
_output_shapes
:(*)
_class
loc:@conv_3/biases/b_conv_3*
T0
і
(conv_3/convolution_and_non_linear/Conv2DConv2Dconv_2/pool/MaxPoolconv_3/weights/W_conv_3/read*
paddingSAME*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
T0*/
_output_shapes
:џџџџџџџџџ(
­
%conv_3/convolution_and_non_linear/addAdd(conv_3/convolution_and_non_linear/Conv2Dconv_3/biases/b_conv_3/read*
T0*/
_output_shapes
:џџџџџџџџџ(

&conv_3/convolution_and_non_linear/ReluRelu%conv_3/convolution_and_non_linear/add*
T0*/
_output_shapes
:џџџџџџџџџ(
в
conv_3/pool/MaxPoolMaxPool&conv_3/convolution_and_non_linear/Relu*
strides
*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ(*
paddingSAME*
T0*
ksize

f
flatten/Reshape/shapeConst*
valueB"џџџџ 
  *
_output_shapes
:*
dtype0

flatten/ReshapeReshapeconv_3/pool/MaxPoolflatten/Reshape/shape*
Tshape0*(
_output_shapes
:џџџџџџџџџ*
T0
t
#fc_1/weights/truncated_normal/shapeConst*
_output_shapes
:*
dtype0*
valueB" 
  d   
g
"fc_1/weights/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
i
$fc_1/weights/truncated_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=
Е
-fc_1/weights/truncated_normal/TruncatedNormalTruncatedNormal#fc_1/weights/truncated_normal/shape*
_output_shapes
:	d*
seed2 *
dtype0*
T0*

seed 
Ї
!fc_1/weights/truncated_normal/mulMul-fc_1/weights/truncated_normal/TruncatedNormal$fc_1/weights/truncated_normal/stddev*
T0*
_output_shapes
:	d

fc_1/weights/truncated_normalAdd!fc_1/weights/truncated_normal/mul"fc_1/weights/truncated_normal/mean*
T0*
_output_shapes
:	d

fc_1/weights/W_fc1
VariableV2*
shared_name *
dtype0*
shape:	d*
_output_shapes
:	d*
	container 
а
fc_1/weights/W_fc1/AssignAssignfc_1/weights/W_fc1fc_1/weights/truncated_normal*
use_locking(*
T0*%
_class
loc:@fc_1/weights/W_fc1*
validate_shape(*
_output_shapes
:	d

fc_1/weights/W_fc1/readIdentityfc_1/weights/W_fc1*
T0*
_output_shapes
:	d*%
_class
loc:@fc_1/weights/W_fc1
^
fc_1/biases/ConstConst*
valueBd*ЭЬЬ=*
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
М
fc_1/biases/b_fc1/AssignAssignfc_1/biases/b_fc1fc_1/biases/Const*
use_locking(*
T0*$
_class
loc:@fc_1/biases/b_fc1*
validate_shape(*
_output_shapes
:d

fc_1/biases/b_fc1/readIdentityfc_1/biases/b_fc1*$
_class
loc:@fc_1/biases/b_fc1*
_output_shapes
:d*
T0
­
!fc_1/matmul_and_non_linear/MatMulMatMulflatten/Reshapefc_1/weights/W_fc1/read*
transpose_b( *'
_output_shapes
:џџџџџџџџџd*
transpose_a( *
T0

fc_1/matmul_and_non_linear/addAdd!fc_1/matmul_and_non_linear/MatMulfc_1/biases/b_fc1/read*
T0*'
_output_shapes
:џџџџџџџџџd
y
fc_1/matmul_and_non_linear/ReluRelufc_1/matmul_and_non_linear/add*'
_output_shapes
:џџџџџџџџџd*
T0
t
#fc_2/weights/truncated_normal/shapeConst*
valueB"d      *
dtype0*
_output_shapes
:
g
"fc_2/weights/truncated_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
i
$fc_2/weights/truncated_normal/stddevConst*
valueB
 *ЭЬЬ=*
_output_shapes
: *
dtype0
Д
-fc_2/weights/truncated_normal/TruncatedNormalTruncatedNormal#fc_2/weights/truncated_normal/shape*
_output_shapes

:d*
seed2 *
T0*

seed *
dtype0
І
!fc_2/weights/truncated_normal/mulMul-fc_2/weights/truncated_normal/TruncatedNormal$fc_2/weights/truncated_normal/stddev*
_output_shapes

:d*
T0

fc_2/weights/truncated_normalAdd!fc_2/weights/truncated_normal/mul"fc_2/weights/truncated_normal/mean*
T0*
_output_shapes

:d

fc_2/weights/W_fc2
VariableV2*
_output_shapes

:d*
	container *
dtype0*
shared_name *
shape
:d
Я
fc_2/weights/W_fc2/AssignAssignfc_2/weights/W_fc2fc_2/weights/truncated_normal*
_output_shapes

:d*
validate_shape(*%
_class
loc:@fc_2/weights/W_fc2*
T0*
use_locking(

fc_2/weights/W_fc2/readIdentityfc_2/weights/W_fc2*%
_class
loc:@fc_2/weights/W_fc2*
_output_shapes

:d*
T0
^
fc_2/biases/ConstConst*
_output_shapes
:*
dtype0*
valueB*ЭЬЬ=
~
fc_2/biases/b_fc_2
VariableV2*
_output_shapes
:*
	container *
shape:*
dtype0*
shared_name 
П
fc_2/biases/b_fc_2/AssignAssignfc_2/biases/b_fc_2fc_2/biases/Const*
_output_shapes
:*
validate_shape(*%
_class
loc:@fc_2/biases/b_fc_2*
T0*
use_locking(

fc_2/biases/b_fc_2/readIdentityfc_2/biases/b_fc_2*
_output_shapes
:*%
_class
loc:@fc_2/biases/b_fc_2*
T0
Њ
outputs/MatMulMatMulfc_1/matmul_and_non_linear/Relufc_2/weights/W_fc2/read*
transpose_b( *'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
T0
m
outputs/addAddoutputs/MatMulfc_2/biases/b_fc_2/read*'
_output_shapes
:џџџџџџџџџ*
T0
Y
outputs/SoftmaxSoftmaxoutputs/add*
T0*'
_output_shapes
:џџџџџџџџџ
[
cross_entorpy/LogLogoutputs/Softmax*
T0*'
_output_shapes
:џџџџџџџџџ
j
cross_entorpy/mulMulinputs/typecross_entorpy/Log*
T0*'
_output_shapes
:џџџџџџџџџ
m
#cross_entorpy/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:

cross_entorpy/SumSumcross_entorpy/mul#cross_entorpy/Sum/reduction_indices*#
_output_shapes
:џџџџџџџџџ*
T0*
	keep_dims( *

Tidx0
Y
cross_entorpy/NegNegcross_entorpy/Sum*
T0*#
_output_shapes
:џџџџџџџџџ
]
cross_entorpy/ConstConst*
valueB: *
dtype0*
_output_shapes
:

cross_entorpy/MeanMeancross_entorpy/Negcross_entorpy/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
|
 cross_entorpy/cross_entorpy/tagsConst*
dtype0*
_output_shapes
: *,
value#B! Bcross_entorpy/cross_entorpy

cross_entorpy/cross_entorpyScalarSummary cross_entorpy/cross_entorpy/tagscross_entorpy/Mean*
_output_shapes
: *
T0
X
train/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Z
train/gradients/ConstConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
k
train/gradients/FillFilltrain/gradients/Shapetrain/gradients/Const*
T0*
_output_shapes
: 

5train/gradients/cross_entorpy/Mean_grad/Reshape/shapeConst*
valueB:*
_output_shapes
:*
dtype0
К
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
д
,train/gradients/cross_entorpy/Mean_grad/TileTile/train/gradients/cross_entorpy/Mean_grad/Reshape-train/gradients/cross_entorpy/Mean_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:џџџџџџџџџ

/train/gradients/cross_entorpy/Mean_grad/Shape_1Shapecross_entorpy/Neg*
T0*
out_type0*
_output_shapes
:
r
/train/gradients/cross_entorpy/Mean_grad/Shape_2Const*
valueB *
_output_shapes
: *
dtype0
w
-train/gradients/cross_entorpy/Mean_grad/ConstConst*
valueB: *
_output_shapes
:*
dtype0
в
,train/gradients/cross_entorpy/Mean_grad/ProdProd/train/gradients/cross_entorpy/Mean_grad/Shape_1-train/gradients/cross_entorpy/Mean_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
y
/train/gradients/cross_entorpy/Mean_grad/Const_1Const*
valueB: *
_output_shapes
:*
dtype0
ж
.train/gradients/cross_entorpy/Mean_grad/Prod_1Prod/train/gradients/cross_entorpy/Mean_grad/Shape_2/train/gradients/cross_entorpy/Mean_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
s
1train/gradients/cross_entorpy/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
О
/train/gradients/cross_entorpy/Mean_grad/MaximumMaximum.train/gradients/cross_entorpy/Mean_grad/Prod_11train/gradients/cross_entorpy/Mean_grad/Maximum/y*
_output_shapes
: *
T0
М
0train/gradients/cross_entorpy/Mean_grad/floordivFloorDiv,train/gradients/cross_entorpy/Mean_grad/Prod/train/gradients/cross_entorpy/Mean_grad/Maximum*
T0*
_output_shapes
: 

,train/gradients/cross_entorpy/Mean_grad/CastCast0train/gradients/cross_entorpy/Mean_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0
Ф
/train/gradients/cross_entorpy/Mean_grad/truedivRealDiv,train/gradients/cross_entorpy/Mean_grad/Tile,train/gradients/cross_entorpy/Mean_grad/Cast*#
_output_shapes
:џџџџџџџџџ*
T0

*train/gradients/cross_entorpy/Neg_grad/NegNeg/train/gradients/cross_entorpy/Mean_grad/truediv*#
_output_shapes
:џџџџџџџџџ*
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
Ј
*train/gradients/cross_entorpy/Sum_grad/addAdd#cross_entorpy/Sum/reduction_indices+train/gradients/cross_entorpy/Sum_grad/Size*
_output_shapes
:*
T0
Д
*train/gradients/cross_entorpy/Sum_grad/modFloorMod*train/gradients/cross_entorpy/Sum_grad/add+train/gradients/cross_entorpy/Sum_grad/Size*
T0*
_output_shapes
:
x
.train/gradients/cross_entorpy/Sum_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
t
2train/gradients/cross_entorpy/Sum_grad/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
t
2train/gradients/cross_entorpy/Sum_grad/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
ђ
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
Л
+train/gradients/cross_entorpy/Sum_grad/FillFill.train/gradients/cross_entorpy/Sum_grad/Shape_11train/gradients/cross_entorpy/Sum_grad/Fill/value*
_output_shapes
:*
T0
Б
4train/gradients/cross_entorpy/Sum_grad/DynamicStitchDynamicStitch,train/gradients/cross_entorpy/Sum_grad/range*train/gradients/cross_entorpy/Sum_grad/mod,train/gradients/cross_entorpy/Sum_grad/Shape+train/gradients/cross_entorpy/Sum_grad/Fill*#
_output_shapes
:џџџџџџџџџ*
T0*
N
r
0train/gradients/cross_entorpy/Sum_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
Я
.train/gradients/cross_entorpy/Sum_grad/MaximumMaximum4train/gradients/cross_entorpy/Sum_grad/DynamicStitch0train/gradients/cross_entorpy/Sum_grad/Maximum/y*
T0*#
_output_shapes
:џџџџџџџџџ
О
/train/gradients/cross_entorpy/Sum_grad/floordivFloorDiv,train/gradients/cross_entorpy/Sum_grad/Shape.train/gradients/cross_entorpy/Sum_grad/Maximum*
_output_shapes
:*
T0
Ь
.train/gradients/cross_entorpy/Sum_grad/ReshapeReshape*train/gradients/cross_entorpy/Neg_grad/Neg4train/gradients/cross_entorpy/Sum_grad/DynamicStitch*
Tshape0*
_output_shapes
:*
T0
и
+train/gradients/cross_entorpy/Sum_grad/TileTile.train/gradients/cross_entorpy/Sum_grad/Reshape/train/gradients/cross_entorpy/Sum_grad/floordiv*

Tmultiples0*
T0*'
_output_shapes
:џџџџџџџџџ
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
№
<train/gradients/cross_entorpy/mul_grad/BroadcastGradientArgsBroadcastGradientArgs,train/gradients/cross_entorpy/mul_grad/Shape.train/gradients/cross_entorpy/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ѓ
*train/gradients/cross_entorpy/mul_grad/mulMul+train/gradients/cross_entorpy/Sum_grad/Tilecross_entorpy/Log*
T0*'
_output_shapes
:џџџџџџџџџ
л
*train/gradients/cross_entorpy/mul_grad/SumSum*train/gradients/cross_entorpy/mul_grad/mul<train/gradients/cross_entorpy/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
г
.train/gradients/cross_entorpy/mul_grad/ReshapeReshape*train/gradients/cross_entorpy/mul_grad/Sum,train/gradients/cross_entorpy/mul_grad/Shape*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0

,train/gradients/cross_entorpy/mul_grad/mul_1Mulinputs/type+train/gradients/cross_entorpy/Sum_grad/Tile*
T0*'
_output_shapes
:џџџџџџџџџ
с
,train/gradients/cross_entorpy/mul_grad/Sum_1Sum,train/gradients/cross_entorpy/mul_grad/mul_1>train/gradients/cross_entorpy/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
й
0train/gradients/cross_entorpy/mul_grad/Reshape_1Reshape,train/gradients/cross_entorpy/mul_grad/Sum_1.train/gradients/cross_entorpy/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Ѓ
7train/gradients/cross_entorpy/mul_grad/tuple/group_depsNoOp/^train/gradients/cross_entorpy/mul_grad/Reshape1^train/gradients/cross_entorpy/mul_grad/Reshape_1
Њ
?train/gradients/cross_entorpy/mul_grad/tuple/control_dependencyIdentity.train/gradients/cross_entorpy/mul_grad/Reshape8^train/gradients/cross_entorpy/mul_grad/tuple/group_deps*A
_class7
53loc:@train/gradients/cross_entorpy/mul_grad/Reshape*'
_output_shapes
:џџџџџџџџџ*
T0
А
Atrain/gradients/cross_entorpy/mul_grad/tuple/control_dependency_1Identity0train/gradients/cross_entorpy/mul_grad/Reshape_18^train/gradients/cross_entorpy/mul_grad/tuple/group_deps*C
_class9
75loc:@train/gradients/cross_entorpy/mul_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ*
T0
Ц
1train/gradients/cross_entorpy/Log_grad/Reciprocal
Reciprocaloutputs/SoftmaxB^train/gradients/cross_entorpy/mul_grad/tuple/control_dependency_1*'
_output_shapes
:џџџџџџџџџ*
T0
й
*train/gradients/cross_entorpy/Log_grad/mulMulAtrain/gradients/cross_entorpy/mul_grad/tuple/control_dependency_11train/gradients/cross_entorpy/Log_grad/Reciprocal*
T0*'
_output_shapes
:џџџџџџџџџ

(train/gradients/outputs/Softmax_grad/mulMul*train/gradients/cross_entorpy/Log_grad/muloutputs/Softmax*
T0*'
_output_shapes
:џџџџџџџџџ

:train/gradients/outputs/Softmax_grad/Sum/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
р
(train/gradients/outputs/Softmax_grad/SumSum(train/gradients/outputs/Softmax_grad/mul:train/gradients/outputs/Softmax_grad/Sum/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:џџџџџџџџџ

2train/gradients/outputs/Softmax_grad/Reshape/shapeConst*
valueB"џџџџ   *
_output_shapes
:*
dtype0
е
,train/gradients/outputs/Softmax_grad/ReshapeReshape(train/gradients/outputs/Softmax_grad/Sum2train/gradients/outputs/Softmax_grad/Reshape/shape*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0
Л
(train/gradients/outputs/Softmax_grad/subSub*train/gradients/cross_entorpy/Log_grad/mul,train/gradients/outputs/Softmax_grad/Reshape*
T0*'
_output_shapes
:џџџџџџџџџ

*train/gradients/outputs/Softmax_grad/mul_1Mul(train/gradients/outputs/Softmax_grad/suboutputs/Softmax*
T0*'
_output_shapes
:џџџџџџџџџ
t
&train/gradients/outputs/add_grad/ShapeShapeoutputs/MatMul*
out_type0*
_output_shapes
:*
T0
r
(train/gradients/outputs/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
о
6train/gradients/outputs/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/outputs/add_grad/Shape(train/gradients/outputs/add_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Я
$train/gradients/outputs/add_grad/SumSum*train/gradients/outputs/Softmax_grad/mul_16train/gradients/outputs/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
С
(train/gradients/outputs/add_grad/ReshapeReshape$train/gradients/outputs/add_grad/Sum&train/gradients/outputs/add_grad/Shape*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0
г
&train/gradients/outputs/add_grad/Sum_1Sum*train/gradients/outputs/Softmax_grad/mul_18train/gradients/outputs/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
К
*train/gradients/outputs/add_grad/Reshape_1Reshape&train/gradients/outputs/add_grad/Sum_1(train/gradients/outputs/add_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0

1train/gradients/outputs/add_grad/tuple/group_depsNoOp)^train/gradients/outputs/add_grad/Reshape+^train/gradients/outputs/add_grad/Reshape_1

9train/gradients/outputs/add_grad/tuple/control_dependencyIdentity(train/gradients/outputs/add_grad/Reshape2^train/gradients/outputs/add_grad/tuple/group_deps*;
_class1
/-loc:@train/gradients/outputs/add_grad/Reshape*'
_output_shapes
:џџџџџџџџџ*
T0

;train/gradients/outputs/add_grad/tuple/control_dependency_1Identity*train/gradients/outputs/add_grad/Reshape_12^train/gradients/outputs/add_grad/tuple/group_deps*=
_class3
1/loc:@train/gradients/outputs/add_grad/Reshape_1*
_output_shapes
:*
T0
р
*train/gradients/outputs/MatMul_grad/MatMulMatMul9train/gradients/outputs/add_grad/tuple/control_dependencyfc_2/weights/W_fc2/read*
transpose_b(*
T0*'
_output_shapes
:џџџџџџџџџd*
transpose_a( 
с
,train/gradients/outputs/MatMul_grad/MatMul_1MatMulfc_1/matmul_and_non_linear/Relu9train/gradients/outputs/add_grad/tuple/control_dependency*
transpose_b( *
_output_shapes

:d*
transpose_a(*
T0

4train/gradients/outputs/MatMul_grad/tuple/group_depsNoOp+^train/gradients/outputs/MatMul_grad/MatMul-^train/gradients/outputs/MatMul_grad/MatMul_1

<train/gradients/outputs/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/outputs/MatMul_grad/MatMul5^train/gradients/outputs/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/outputs/MatMul_grad/MatMul*'
_output_shapes
:џџџџџџџџџd

>train/gradients/outputs/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/outputs/MatMul_grad/MatMul_15^train/gradients/outputs/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/outputs/MatMul_grad/MatMul_1*
_output_shapes

:d
к
=train/gradients/fc_1/matmul_and_non_linear/Relu_grad/ReluGradReluGrad<train/gradients/outputs/MatMul_grad/tuple/control_dependencyfc_1/matmul_and_non_linear/Relu*'
_output_shapes
:џџџџџџџџџd*
T0

9train/gradients/fc_1/matmul_and_non_linear/add_grad/ShapeShape!fc_1/matmul_and_non_linear/MatMul*
T0*
out_type0*
_output_shapes
:

;train/gradients/fc_1/matmul_and_non_linear/add_grad/Shape_1Const*
valueB:d*
dtype0*
_output_shapes
:

Itrain/gradients/fc_1/matmul_and_non_linear/add_grad/BroadcastGradientArgsBroadcastGradientArgs9train/gradients/fc_1/matmul_and_non_linear/add_grad/Shape;train/gradients/fc_1/matmul_and_non_linear/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

7train/gradients/fc_1/matmul_and_non_linear/add_grad/SumSum=train/gradients/fc_1/matmul_and_non_linear/Relu_grad/ReluGradItrain/gradients/fc_1/matmul_and_non_linear/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
њ
;train/gradients/fc_1/matmul_and_non_linear/add_grad/ReshapeReshape7train/gradients/fc_1/matmul_and_non_linear/add_grad/Sum9train/gradients/fc_1/matmul_and_non_linear/add_grad/Shape*
Tshape0*'
_output_shapes
:џџџџџџџџџd*
T0

9train/gradients/fc_1/matmul_and_non_linear/add_grad/Sum_1Sum=train/gradients/fc_1/matmul_and_non_linear/Relu_grad/ReluGradKtrain/gradients/fc_1/matmul_and_non_linear/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
ѓ
=train/gradients/fc_1/matmul_and_non_linear/add_grad/Reshape_1Reshape9train/gradients/fc_1/matmul_and_non_linear/add_grad/Sum_1;train/gradients/fc_1/matmul_and_non_linear/add_grad/Shape_1*
Tshape0*
_output_shapes
:d*
T0
Ъ
Dtrain/gradients/fc_1/matmul_and_non_linear/add_grad/tuple/group_depsNoOp<^train/gradients/fc_1/matmul_and_non_linear/add_grad/Reshape>^train/gradients/fc_1/matmul_and_non_linear/add_grad/Reshape_1
о
Ltrain/gradients/fc_1/matmul_and_non_linear/add_grad/tuple/control_dependencyIdentity;train/gradients/fc_1/matmul_and_non_linear/add_grad/ReshapeE^train/gradients/fc_1/matmul_and_non_linear/add_grad/tuple/group_deps*N
_classD
B@loc:@train/gradients/fc_1/matmul_and_non_linear/add_grad/Reshape*'
_output_shapes
:џџџџџџџџџd*
T0
з
Ntrain/gradients/fc_1/matmul_and_non_linear/add_grad/tuple/control_dependency_1Identity=train/gradients/fc_1/matmul_and_non_linear/add_grad/Reshape_1E^train/gradients/fc_1/matmul_and_non_linear/add_grad/tuple/group_deps*P
_classF
DBloc:@train/gradients/fc_1/matmul_and_non_linear/add_grad/Reshape_1*
_output_shapes
:d*
T0

=train/gradients/fc_1/matmul_and_non_linear/MatMul_grad/MatMulMatMulLtrain/gradients/fc_1/matmul_and_non_linear/add_grad/tuple/control_dependencyfc_1/weights/W_fc1/read*
transpose_b(*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
T0
ј
?train/gradients/fc_1/matmul_and_non_linear/MatMul_grad/MatMul_1MatMulflatten/ReshapeLtrain/gradients/fc_1/matmul_and_non_linear/add_grad/tuple/control_dependency*
transpose_b( *
_output_shapes
:	d*
transpose_a(*
T0
б
Gtrain/gradients/fc_1/matmul_and_non_linear/MatMul_grad/tuple/group_depsNoOp>^train/gradients/fc_1/matmul_and_non_linear/MatMul_grad/MatMul@^train/gradients/fc_1/matmul_and_non_linear/MatMul_grad/MatMul_1
щ
Otrain/gradients/fc_1/matmul_and_non_linear/MatMul_grad/tuple/control_dependencyIdentity=train/gradients/fc_1/matmul_and_non_linear/MatMul_grad/MatMulH^train/gradients/fc_1/matmul_and_non_linear/MatMul_grad/tuple/group_deps*P
_classF
DBloc:@train/gradients/fc_1/matmul_and_non_linear/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ*
T0
ц
Qtrain/gradients/fc_1/matmul_and_non_linear/MatMul_grad/tuple/control_dependency_1Identity?train/gradients/fc_1/matmul_and_non_linear/MatMul_grad/MatMul_1H^train/gradients/fc_1/matmul_and_non_linear/MatMul_grad/tuple/group_deps*
T0*R
_classH
FDloc:@train/gradients/fc_1/matmul_and_non_linear/MatMul_grad/MatMul_1*
_output_shapes
:	d
}
*train/gradients/flatten/Reshape_grad/ShapeShapeconv_3/pool/MaxPool*
out_type0*
_output_shapes
:*
T0
ќ
,train/gradients/flatten/Reshape_grad/ReshapeReshapeOtrain/gradients/fc_1/matmul_and_non_linear/MatMul_grad/tuple/control_dependency*train/gradients/flatten/Reshape_grad/Shape*
Tshape0*/
_output_shapes
:џџџџџџџџџ(*
T0
К
4train/gradients/conv_3/pool/MaxPool_grad/MaxPoolGradMaxPoolGrad&conv_3/convolution_and_non_linear/Reluconv_3/pool/MaxPool,train/gradients/flatten/Reshape_grad/Reshape*
ksize
*/
_output_shapes
:џџџџџџџџџ(*
T0*
data_formatNHWC*
strides
*
paddingSAME
ш
Dtrain/gradients/conv_3/convolution_and_non_linear/Relu_grad/ReluGradReluGrad4train/gradients/conv_3/pool/MaxPool_grad/MaxPoolGrad&conv_3/convolution_and_non_linear/Relu*/
_output_shapes
:џџџџџџџџџ(*
T0
Ј
@train/gradients/conv_3/convolution_and_non_linear/add_grad/ShapeShape(conv_3/convolution_and_non_linear/Conv2D*
T0*
out_type0*
_output_shapes
:

Btrain/gradients/conv_3/convolution_and_non_linear/add_grad/Shape_1Const*
valueB:(*
dtype0*
_output_shapes
:
Ќ
Ptrain/gradients/conv_3/convolution_and_non_linear/add_grad/BroadcastGradientArgsBroadcastGradientArgs@train/gradients/conv_3/convolution_and_non_linear/add_grad/ShapeBtrain/gradients/conv_3/convolution_and_non_linear/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

>train/gradients/conv_3/convolution_and_non_linear/add_grad/SumSumDtrain/gradients/conv_3/convolution_and_non_linear/Relu_grad/ReluGradPtrain/gradients/conv_3/convolution_and_non_linear/add_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0

Btrain/gradients/conv_3/convolution_and_non_linear/add_grad/ReshapeReshape>train/gradients/conv_3/convolution_and_non_linear/add_grad/Sum@train/gradients/conv_3/convolution_and_non_linear/add_grad/Shape*
Tshape0*/
_output_shapes
:џџџџџџџџџ(*
T0
Ё
@train/gradients/conv_3/convolution_and_non_linear/add_grad/Sum_1SumDtrain/gradients/conv_3/convolution_and_non_linear/Relu_grad/ReluGradRtrain/gradients/conv_3/convolution_and_non_linear/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

Dtrain/gradients/conv_3/convolution_and_non_linear/add_grad/Reshape_1Reshape@train/gradients/conv_3/convolution_and_non_linear/add_grad/Sum_1Btrain/gradients/conv_3/convolution_and_non_linear/add_grad/Shape_1*
Tshape0*
_output_shapes
:(*
T0
п
Ktrain/gradients/conv_3/convolution_and_non_linear/add_grad/tuple/group_depsNoOpC^train/gradients/conv_3/convolution_and_non_linear/add_grad/ReshapeE^train/gradients/conv_3/convolution_and_non_linear/add_grad/Reshape_1

Strain/gradients/conv_3/convolution_and_non_linear/add_grad/tuple/control_dependencyIdentityBtrain/gradients/conv_3/convolution_and_non_linear/add_grad/ReshapeL^train/gradients/conv_3/convolution_and_non_linear/add_grad/tuple/group_deps*U
_classK
IGloc:@train/gradients/conv_3/convolution_and_non_linear/add_grad/Reshape*/
_output_shapes
:џџџџџџџџџ(*
T0
ѓ
Utrain/gradients/conv_3/convolution_and_non_linear/add_grad/tuple/control_dependency_1IdentityDtrain/gradients/conv_3/convolution_and_non_linear/add_grad/Reshape_1L^train/gradients/conv_3/convolution_and_non_linear/add_grad/tuple/group_deps*W
_classM
KIloc:@train/gradients/conv_3/convolution_and_non_linear/add_grad/Reshape_1*
_output_shapes
:(*
T0

Ctrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/ShapeShapeconv_2/pool/MaxPool*
out_type0*
_output_shapes
:*
T0
Ь
Qtrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputCtrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/Shapeconv_3/weights/W_conv_3/readStrain/gradients/conv_3/convolution_and_non_linear/add_grad/tuple/control_dependency*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingSAME

Etrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/Shape_1Const*%
valueB"         (   *
dtype0*
_output_shapes
:
Ѓ
Rtrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterconv_2/pool/MaxPoolEtrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/Shape_1Strain/gradients/conv_3/convolution_and_non_linear/add_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
:(
џ
Ntrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/tuple/group_depsNoOpR^train/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInputS^train/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilter
І
Vtrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/tuple/control_dependencyIdentityQtrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInputO^train/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@train/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:џџџџџџџџџ
Ё
Xtrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/tuple/control_dependency_1IdentityRtrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilterO^train/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/tuple/group_deps*e
_class[
YWloc:@train/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:(*
T0
ф
4train/gradients/conv_2/pool/MaxPool_grad/MaxPoolGradMaxPoolGrad&conv_2/convolution_and_non_linear/Reluconv_2/pool/MaxPoolVtrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/tuple/control_dependency*
paddingSAME*
T0*
data_formatNHWC*
strides
*/
_output_shapes
:џџџџџџџџџ  *
ksize

ш
Dtrain/gradients/conv_2/convolution_and_non_linear/Relu_grad/ReluGradReluGrad4train/gradients/conv_2/pool/MaxPool_grad/MaxPoolGrad&conv_2/convolution_and_non_linear/Relu*/
_output_shapes
:џџџџџџџџџ  *
T0
Ј
@train/gradients/conv_2/convolution_and_non_linear/add_grad/ShapeShape(conv_2/convolution_and_non_linear/Conv2D*
out_type0*
_output_shapes
:*
T0

Btrain/gradients/conv_2/convolution_and_non_linear/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
Ќ
Ptrain/gradients/conv_2/convolution_and_non_linear/add_grad/BroadcastGradientArgsBroadcastGradientArgs@train/gradients/conv_2/convolution_and_non_linear/add_grad/ShapeBtrain/gradients/conv_2/convolution_and_non_linear/add_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

>train/gradients/conv_2/convolution_and_non_linear/add_grad/SumSumDtrain/gradients/conv_2/convolution_and_non_linear/Relu_grad/ReluGradPtrain/gradients/conv_2/convolution_and_non_linear/add_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0

Btrain/gradients/conv_2/convolution_and_non_linear/add_grad/ReshapeReshape>train/gradients/conv_2/convolution_and_non_linear/add_grad/Sum@train/gradients/conv_2/convolution_and_non_linear/add_grad/Shape*
Tshape0*/
_output_shapes
:џџџџџџџџџ  *
T0
Ё
@train/gradients/conv_2/convolution_and_non_linear/add_grad/Sum_1SumDtrain/gradients/conv_2/convolution_and_non_linear/Relu_grad/ReluGradRtrain/gradients/conv_2/convolution_and_non_linear/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

Dtrain/gradients/conv_2/convolution_and_non_linear/add_grad/Reshape_1Reshape@train/gradients/conv_2/convolution_and_non_linear/add_grad/Sum_1Btrain/gradients/conv_2/convolution_and_non_linear/add_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
п
Ktrain/gradients/conv_2/convolution_and_non_linear/add_grad/tuple/group_depsNoOpC^train/gradients/conv_2/convolution_and_non_linear/add_grad/ReshapeE^train/gradients/conv_2/convolution_and_non_linear/add_grad/Reshape_1

Strain/gradients/conv_2/convolution_and_non_linear/add_grad/tuple/control_dependencyIdentityBtrain/gradients/conv_2/convolution_and_non_linear/add_grad/ReshapeL^train/gradients/conv_2/convolution_and_non_linear/add_grad/tuple/group_deps*U
_classK
IGloc:@train/gradients/conv_2/convolution_and_non_linear/add_grad/Reshape*/
_output_shapes
:џџџџџџџџџ  *
T0
ѓ
Utrain/gradients/conv_2/convolution_and_non_linear/add_grad/tuple/control_dependency_1IdentityDtrain/gradients/conv_2/convolution_and_non_linear/add_grad/Reshape_1L^train/gradients/conv_2/convolution_and_non_linear/add_grad/tuple/group_deps*
T0*W
_classM
KIloc:@train/gradients/conv_2/convolution_and_non_linear/add_grad/Reshape_1*
_output_shapes
:

Ctrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/ShapeShapeconv_1/pool/MaxPool*
T0*
out_type0*
_output_shapes
:
Ь
Qtrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputCtrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/Shapeconv_2/weights/W_conv_2/readStrain/gradients/conv_2/convolution_and_non_linear/add_grad/tuple/control_dependency*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingSAME

Etrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/Shape_1Const*%
valueB"      
      *
_output_shapes
:*
dtype0
Ѓ
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
џ
Ntrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/tuple/group_depsNoOpR^train/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInputS^train/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilter
І
Vtrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/tuple/control_dependencyIdentityQtrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInputO^train/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@train/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:џџџџџџџџџ  

Ё
Xtrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/tuple/control_dependency_1IdentityRtrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilterO^train/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/tuple/group_deps*
T0*e
_class[
YWloc:@train/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:

ц
4train/gradients/conv_1/pool/MaxPool_grad/MaxPoolGradMaxPoolGrad&conv_1/convolution_and_non_linear/Reluconv_1/pool/MaxPoolVtrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/tuple/control_dependency*
ksize
*
T0*
paddingSAME*1
_output_shapes
:џџџџџџџџџ
*
data_formatNHWC*
strides

ъ
Dtrain/gradients/conv_1/convolution_and_non_linear/Relu_grad/ReluGradReluGrad4train/gradients/conv_1/pool/MaxPool_grad/MaxPoolGrad&conv_1/convolution_and_non_linear/Relu*
T0*1
_output_shapes
:џџџџџџџџџ

Ј
@train/gradients/conv_1/convolution_and_non_linear/add_grad/ShapeShape(conv_1/convolution_and_non_linear/Conv2D*
T0*
out_type0*
_output_shapes
:

Btrain/gradients/conv_1/convolution_and_non_linear/add_grad/Shape_1Const*
valueB:
*
_output_shapes
:*
dtype0
Ќ
Ptrain/gradients/conv_1/convolution_and_non_linear/add_grad/BroadcastGradientArgsBroadcastGradientArgs@train/gradients/conv_1/convolution_and_non_linear/add_grad/ShapeBtrain/gradients/conv_1/convolution_and_non_linear/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

>train/gradients/conv_1/convolution_and_non_linear/add_grad/SumSumDtrain/gradients/conv_1/convolution_and_non_linear/Relu_grad/ReluGradPtrain/gradients/conv_1/convolution_and_non_linear/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

Btrain/gradients/conv_1/convolution_and_non_linear/add_grad/ReshapeReshape>train/gradients/conv_1/convolution_and_non_linear/add_grad/Sum@train/gradients/conv_1/convolution_and_non_linear/add_grad/Shape*
Tshape0*1
_output_shapes
:џџџџџџџџџ
*
T0
Ё
@train/gradients/conv_1/convolution_and_non_linear/add_grad/Sum_1SumDtrain/gradients/conv_1/convolution_and_non_linear/Relu_grad/ReluGradRtrain/gradients/conv_1/convolution_and_non_linear/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

Dtrain/gradients/conv_1/convolution_and_non_linear/add_grad/Reshape_1Reshape@train/gradients/conv_1/convolution_and_non_linear/add_grad/Sum_1Btrain/gradients/conv_1/convolution_and_non_linear/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:

п
Ktrain/gradients/conv_1/convolution_and_non_linear/add_grad/tuple/group_depsNoOpC^train/gradients/conv_1/convolution_and_non_linear/add_grad/ReshapeE^train/gradients/conv_1/convolution_and_non_linear/add_grad/Reshape_1

Strain/gradients/conv_1/convolution_and_non_linear/add_grad/tuple/control_dependencyIdentityBtrain/gradients/conv_1/convolution_and_non_linear/add_grad/ReshapeL^train/gradients/conv_1/convolution_and_non_linear/add_grad/tuple/group_deps*
T0*U
_classK
IGloc:@train/gradients/conv_1/convolution_and_non_linear/add_grad/Reshape*1
_output_shapes
:џџџџџџџџџ

ѓ
Utrain/gradients/conv_1/convolution_and_non_linear/add_grad/tuple/control_dependency_1IdentityDtrain/gradients/conv_1/convolution_and_non_linear/add_grad/Reshape_1L^train/gradients/conv_1/convolution_and_non_linear/add_grad/tuple/group_deps*
T0*W
_classM
KIloc:@train/gradients/conv_1/convolution_and_non_linear/add_grad/Reshape_1*
_output_shapes
:


Ctrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/ShapeShapeprepare_tensors/Reshape*
out_type0*
_output_shapes
:*
T0
Ь
Qtrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputCtrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/Shapeconv_1/weights/W_conv_1/readStrain/gradients/conv_1/convolution_and_non_linear/add_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ

Etrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/Shape_1Const*%
valueB"         
   *
_output_shapes
:*
dtype0
Ї
Rtrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterprepare_tensors/ReshapeEtrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/Shape_1Strain/gradients/conv_1/convolution_and_non_linear/add_grad/tuple/control_dependency*&
_output_shapes
:
*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingSAME
џ
Ntrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/tuple/group_depsNoOpR^train/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInputS^train/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilter
Ј
Vtrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/tuple/control_dependencyIdentityQtrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInputO^train/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@train/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInput*1
_output_shapes
:џџџџџџџџџ
Ё
Xtrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/tuple/control_dependency_1IdentityRtrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilterO^train/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/tuple/group_deps*
T0*e
_class[
YWloc:@train/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:


train/beta1_power/initial_valueConst*
valueB
 *fff?**
_class 
loc:@conv_1/weights/W_conv_1*
_output_shapes
: *
dtype0
Ё
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
Ь
train/beta1_power/AssignAssigntrain/beta1_powertrain/beta1_power/initial_value*
use_locking(*
T0**
_class 
loc:@conv_1/weights/W_conv_1*
validate_shape(*
_output_shapes
: 

train/beta1_power/readIdentitytrain/beta1_power**
_class 
loc:@conv_1/weights/W_conv_1*
_output_shapes
: *
T0

train/beta2_power/initial_valueConst*
valueB
 *wО?**
_class 
loc:@conv_1/weights/W_conv_1*
_output_shapes
: *
dtype0
Ё
train/beta2_power
VariableV2*
shared_name **
_class 
loc:@conv_1/weights/W_conv_1*
	container *
shape: *
dtype0*
_output_shapes
: 
Ь
train/beta2_power/AssignAssigntrain/beta2_powertrain/beta2_power/initial_value*
use_locking(*
T0**
_class 
loc:@conv_1/weights/W_conv_1*
validate_shape(*
_output_shapes
: 

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

Ь
conv_1/weights/W_conv_1/Adam
VariableV2**
_class 
loc:@conv_1/weights/W_conv_1*&
_output_shapes
:
*
shape:
*
dtype0*
shared_name *
	container 
о
#conv_1/weights/W_conv_1/Adam/AssignAssignconv_1/weights/W_conv_1/Adamtrain/zeros**
_class 
loc:@conv_1/weights/W_conv_1*&
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
Ј
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
Ю
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
ф
%conv_1/weights/W_conv_1/Adam_1/AssignAssignconv_1/weights/W_conv_1/Adam_1train/zeros_1**
_class 
loc:@conv_1/weights/W_conv_1*&
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
Ќ
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

В
conv_1/biases/b_conv_1/Adam
VariableV2*
shared_name *)
_class
loc:@conv_1/biases/b_conv_1*
	container *
shape:
*
dtype0*
_output_shapes
:

б
"conv_1/biases/b_conv_1/Adam/AssignAssignconv_1/biases/b_conv_1/Adamtrain/zeros_2*
use_locking(*
T0*)
_class
loc:@conv_1/biases/b_conv_1*
validate_shape(*
_output_shapes
:


 conv_1/biases/b_conv_1/Adam/readIdentityconv_1/biases/b_conv_1/Adam*
T0*)
_class
loc:@conv_1/biases/b_conv_1*
_output_shapes
:

Z
train/zeros_3Const*
valueB
*    *
_output_shapes
:
*
dtype0
Д
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
е
$conv_1/biases/b_conv_1/Adam_1/AssignAssignconv_1/biases/b_conv_1/Adam_1train/zeros_3*
use_locking(*
T0*)
_class
loc:@conv_1/biases/b_conv_1*
validate_shape(*
_output_shapes
:


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
Ь
conv_2/weights/W_conv_2/Adam
VariableV2*
shape:
*&
_output_shapes
:
*
shared_name **
_class 
loc:@conv_2/weights/W_conv_2*
dtype0*
	container 
р
#conv_2/weights/W_conv_2/Adam/AssignAssignconv_2/weights/W_conv_2/Adamtrain/zeros_4*
use_locking(*
T0**
_class 
loc:@conv_2/weights/W_conv_2*
validate_shape(*&
_output_shapes
:

Ј
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
*    *
dtype0*&
_output_shapes
:

Ю
conv_2/weights/W_conv_2/Adam_1
VariableV2*
shape:
*&
_output_shapes
:
*
shared_name **
_class 
loc:@conv_2/weights/W_conv_2*
dtype0*
	container 
ф
%conv_2/weights/W_conv_2/Adam_1/AssignAssignconv_2/weights/W_conv_2/Adam_1train/zeros_5**
_class 
loc:@conv_2/weights/W_conv_2*&
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
Ќ
#conv_2/weights/W_conv_2/Adam_1/readIdentityconv_2/weights/W_conv_2/Adam_1**
_class 
loc:@conv_2/weights/W_conv_2*&
_output_shapes
:
*
T0
Z
train/zeros_6Const*
valueB*    *
dtype0*
_output_shapes
:
В
conv_2/biases/b_conv_2/Adam
VariableV2*
	container *
dtype0*)
_class
loc:@conv_2/biases/b_conv_2*
_output_shapes
:*
shape:*
shared_name 
б
"conv_2/biases/b_conv_2/Adam/AssignAssignconv_2/biases/b_conv_2/Adamtrain/zeros_6*)
_class
loc:@conv_2/biases/b_conv_2*
_output_shapes
:*
T0*
validate_shape(*
use_locking(

 conv_2/biases/b_conv_2/Adam/readIdentityconv_2/biases/b_conv_2/Adam*)
_class
loc:@conv_2/biases/b_conv_2*
_output_shapes
:*
T0
Z
train/zeros_7Const*
valueB*    *
_output_shapes
:*
dtype0
Д
conv_2/biases/b_conv_2/Adam_1
VariableV2*)
_class
loc:@conv_2/biases/b_conv_2*
_output_shapes
:*
shape:*
dtype0*
shared_name *
	container 
е
$conv_2/biases/b_conv_2/Adam_1/AssignAssignconv_2/biases/b_conv_2/Adam_1train/zeros_7*)
_class
loc:@conv_2/biases/b_conv_2*
_output_shapes
:*
T0*
validate_shape(*
use_locking(

"conv_2/biases/b_conv_2/Adam_1/readIdentityconv_2/biases/b_conv_2/Adam_1*)
_class
loc:@conv_2/biases/b_conv_2*
_output_shapes
:*
T0
r
train/zeros_8Const*%
valueB(*    *&
_output_shapes
:(*
dtype0
Ь
conv_3/weights/W_conv_3/Adam
VariableV2**
_class 
loc:@conv_3/weights/W_conv_3*&
_output_shapes
:(*
shape:(*
dtype0*
shared_name *
	container 
р
#conv_3/weights/W_conv_3/Adam/AssignAssignconv_3/weights/W_conv_3/Adamtrain/zeros_8**
_class 
loc:@conv_3/weights/W_conv_3*&
_output_shapes
:(*
T0*
validate_shape(*
use_locking(
Ј
!conv_3/weights/W_conv_3/Adam/readIdentityconv_3/weights/W_conv_3/Adam**
_class 
loc:@conv_3/weights/W_conv_3*&
_output_shapes
:(*
T0
r
train/zeros_9Const*%
valueB(*    *
dtype0*&
_output_shapes
:(
Ю
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
ф
%conv_3/weights/W_conv_3/Adam_1/AssignAssignconv_3/weights/W_conv_3/Adam_1train/zeros_9*
use_locking(*
T0**
_class 
loc:@conv_3/weights/W_conv_3*
validate_shape(*&
_output_shapes
:(
Ќ
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
В
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
в
"conv_3/biases/b_conv_3/Adam/AssignAssignconv_3/biases/b_conv_3/Adamtrain/zeros_10*
use_locking(*
T0*)
_class
loc:@conv_3/biases/b_conv_3*
validate_shape(*
_output_shapes
:(

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
Д
conv_3/biases/b_conv_3/Adam_1
VariableV2*
	container *
dtype0*)
_class
loc:@conv_3/biases/b_conv_3*
_output_shapes
:(*
shape:(*
shared_name 
ж
$conv_3/biases/b_conv_3/Adam_1/AssignAssignconv_3/biases/b_conv_3/Adam_1train/zeros_11*
use_locking(*
T0*)
_class
loc:@conv_3/biases/b_conv_3*
validate_shape(*
_output_shapes
:(

"conv_3/biases/b_conv_3/Adam_1/readIdentityconv_3/biases/b_conv_3/Adam_1*
T0*)
_class
loc:@conv_3/biases/b_conv_3*
_output_shapes
:(
e
train/zeros_12Const*
valueB	d*    *
dtype0*
_output_shapes
:	d
Д
fc_1/weights/W_fc1/Adam
VariableV2*
shared_name *%
_class
loc:@fc_1/weights/W_fc1*
	container *
shape:	d*
dtype0*
_output_shapes
:	d
Ы
fc_1/weights/W_fc1/Adam/AssignAssignfc_1/weights/W_fc1/Adamtrain/zeros_12*
use_locking(*
T0*%
_class
loc:@fc_1/weights/W_fc1*
validate_shape(*
_output_shapes
:	d

fc_1/weights/W_fc1/Adam/readIdentityfc_1/weights/W_fc1/Adam*
T0*%
_class
loc:@fc_1/weights/W_fc1*
_output_shapes
:	d
e
train/zeros_13Const*
valueB	d*    *
dtype0*
_output_shapes
:	d
Ж
fc_1/weights/W_fc1/Adam_1
VariableV2*
shape:	d*
_output_shapes
:	d*
shared_name *%
_class
loc:@fc_1/weights/W_fc1*
dtype0*
	container 
Я
 fc_1/weights/W_fc1/Adam_1/AssignAssignfc_1/weights/W_fc1/Adam_1train/zeros_13*
use_locking(*
T0*%
_class
loc:@fc_1/weights/W_fc1*
validate_shape(*
_output_shapes
:	d

fc_1/weights/W_fc1/Adam_1/readIdentityfc_1/weights/W_fc1/Adam_1*
T0*%
_class
loc:@fc_1/weights/W_fc1*
_output_shapes
:	d
[
train/zeros_14Const*
valueBd*    *
dtype0*
_output_shapes
:d
Ј
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
У
fc_1/biases/b_fc1/Adam/AssignAssignfc_1/biases/b_fc1/Adamtrain/zeros_14*$
_class
loc:@fc_1/biases/b_fc1*
_output_shapes
:d*
T0*
validate_shape(*
use_locking(

fc_1/biases/b_fc1/Adam/readIdentityfc_1/biases/b_fc1/Adam*
T0*$
_class
loc:@fc_1/biases/b_fc1*
_output_shapes
:d
[
train/zeros_15Const*
valueBd*    *
dtype0*
_output_shapes
:d
Њ
fc_1/biases/b_fc1/Adam_1
VariableV2*
shape:d*
_output_shapes
:d*
shared_name *$
_class
loc:@fc_1/biases/b_fc1*
dtype0*
	container 
Ч
fc_1/biases/b_fc1/Adam_1/AssignAssignfc_1/biases/b_fc1/Adam_1train/zeros_15*$
_class
loc:@fc_1/biases/b_fc1*
_output_shapes
:d*
T0*
validate_shape(*
use_locking(

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
В
fc_2/weights/W_fc2/Adam
VariableV2*
	container *
dtype0*%
_class
loc:@fc_2/weights/W_fc2*
_output_shapes

:d*
shape
:d*
shared_name 
Ъ
fc_2/weights/W_fc2/Adam/AssignAssignfc_2/weights/W_fc2/Adamtrain/zeros_16*
use_locking(*
T0*%
_class
loc:@fc_2/weights/W_fc2*
validate_shape(*
_output_shapes

:d

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
Д
fc_2/weights/W_fc2/Adam_1
VariableV2*
	container *
dtype0*%
_class
loc:@fc_2/weights/W_fc2*
_output_shapes

:d*
shape
:d*
shared_name 
Ю
 fc_2/weights/W_fc2/Adam_1/AssignAssignfc_2/weights/W_fc2/Adam_1train/zeros_17*
use_locking(*
T0*%
_class
loc:@fc_2/weights/W_fc2*
validate_shape(*
_output_shapes

:d

fc_2/weights/W_fc2/Adam_1/readIdentityfc_2/weights/W_fc2/Adam_1*%
_class
loc:@fc_2/weights/W_fc2*
_output_shapes

:d*
T0
[
train/zeros_18Const*
valueB*    *
_output_shapes
:*
dtype0
Њ
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
Ц
fc_2/biases/b_fc_2/Adam/AssignAssignfc_2/biases/b_fc_2/Adamtrain/zeros_18*%
_class
loc:@fc_2/biases/b_fc_2*
_output_shapes
:*
T0*
validate_shape(*
use_locking(

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
Ќ
fc_2/biases/b_fc_2/Adam_1
VariableV2*%
_class
loc:@fc_2/biases/b_fc_2*
_output_shapes
:*
shape:*
dtype0*
shared_name *
	container 
Ъ
 fc_2/biases/b_fc_2/Adam_1/AssignAssignfc_2/biases/b_fc_2/Adam_1train/zeros_19*%
_class
loc:@fc_2/biases/b_fc_2*
_output_shapes
:*
T0*
validate_shape(*
use_locking(

fc_2/biases/b_fc_2/Adam_1/readIdentityfc_2/biases/b_fc_2/Adam_1*
T0*%
_class
loc:@fc_2/biases/b_fc_2*
_output_shapes
:
]
train/Adam/learning_rateConst*
valueB
 *Зб8*
dtype0*
_output_shapes
: 
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
 *wО?*
_output_shapes
: *
dtype0
W
train/Adam/epsilonConst*
valueB
 *wЬ+2*
dtype0*
_output_shapes
: 
у
3train/Adam/update_conv_1/weights/W_conv_1/ApplyAdam	ApplyAdamconv_1/weights/W_conv_1conv_1/weights/W_conv_1/Adamconv_1/weights/W_conv_1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonXtrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0**
_class 
loc:@conv_1/weights/W_conv_1*&
_output_shapes
:

Я
2train/Adam/update_conv_1/biases/b_conv_1/ApplyAdam	ApplyAdamconv_1/biases/b_conv_1conv_1/biases/b_conv_1/Adamconv_1/biases/b_conv_1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonUtrain/gradients/conv_1/convolution_and_non_linear/add_grad/tuple/control_dependency_1*)
_class
loc:@conv_1/biases/b_conv_1*
_output_shapes
:
*
T0*
use_locking( 
у
3train/Adam/update_conv_2/weights/W_conv_2/ApplyAdam	ApplyAdamconv_2/weights/W_conv_2conv_2/weights/W_conv_2/Adamconv_2/weights/W_conv_2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonXtrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/tuple/control_dependency_1**
_class 
loc:@conv_2/weights/W_conv_2*&
_output_shapes
:
*
T0*
use_locking( 
Я
2train/Adam/update_conv_2/biases/b_conv_2/ApplyAdam	ApplyAdamconv_2/biases/b_conv_2conv_2/biases/b_conv_2/Adamconv_2/biases/b_conv_2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonUtrain/gradients/conv_2/convolution_and_non_linear/add_grad/tuple/control_dependency_1*)
_class
loc:@conv_2/biases/b_conv_2*
_output_shapes
:*
T0*
use_locking( 
у
3train/Adam/update_conv_3/weights/W_conv_3/ApplyAdam	ApplyAdamconv_3/weights/W_conv_3conv_3/weights/W_conv_3/Adamconv_3/weights/W_conv_3/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonXtrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/tuple/control_dependency_1**
_class 
loc:@conv_3/weights/W_conv_3*&
_output_shapes
:(*
T0*
use_locking( 
Я
2train/Adam/update_conv_3/biases/b_conv_3/ApplyAdam	ApplyAdamconv_3/biases/b_conv_3conv_3/biases/b_conv_3/Adamconv_3/biases/b_conv_3/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonUtrain/gradients/conv_3/convolution_and_non_linear/add_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@conv_3/biases/b_conv_3*
_output_shapes
:(
М
.train/Adam/update_fc_1/weights/W_fc1/ApplyAdam	ApplyAdamfc_1/weights/W_fc1fc_1/weights/W_fc1/Adamfc_1/weights/W_fc1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonQtrain/gradients/fc_1/matmul_and_non_linear/MatMul_grad/tuple/control_dependency_1*%
_class
loc:@fc_1/weights/W_fc1*
_output_shapes
:	d*
T0*
use_locking( 
Џ
-train/Adam/update_fc_1/biases/b_fc1/ApplyAdam	ApplyAdamfc_1/biases/b_fc1fc_1/biases/b_fc1/Adamfc_1/biases/b_fc1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonNtrain/gradients/fc_1/matmul_and_non_linear/add_grad/tuple/control_dependency_1*$
_class
loc:@fc_1/biases/b_fc1*
_output_shapes
:d*
T0*
use_locking( 
Ј
.train/Adam/update_fc_2/weights/W_fc2/ApplyAdam	ApplyAdamfc_2/weights/W_fc2fc_2/weights/W_fc2/Adamfc_2/weights/W_fc2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/outputs/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*%
_class
loc:@fc_2/weights/W_fc2*
_output_shapes

:d
Ё
.train/Adam/update_fc_2/biases/b_fc_2/ApplyAdam	ApplyAdamfc_2/biases/b_fc_2fc_2/biases/b_fc_2/Adamfc_2/biases/b_fc_2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/outputs/add_grad/tuple/control_dependency_1*%
_class
loc:@fc_2/biases/b_fc_2*
_output_shapes
:*
T0*
use_locking( 

train/Adam/mulMultrain/beta1_power/readtrain/Adam/beta14^train/Adam/update_conv_1/weights/W_conv_1/ApplyAdam3^train/Adam/update_conv_1/biases/b_conv_1/ApplyAdam4^train/Adam/update_conv_2/weights/W_conv_2/ApplyAdam3^train/Adam/update_conv_2/biases/b_conv_2/ApplyAdam4^train/Adam/update_conv_3/weights/W_conv_3/ApplyAdam3^train/Adam/update_conv_3/biases/b_conv_3/ApplyAdam/^train/Adam/update_fc_1/weights/W_fc1/ApplyAdam.^train/Adam/update_fc_1/biases/b_fc1/ApplyAdam/^train/Adam/update_fc_2/weights/W_fc2/ApplyAdam/^train/Adam/update_fc_2/biases/b_fc_2/ApplyAdam**
_class 
loc:@conv_1/weights/W_conv_1*
_output_shapes
: *
T0
Д
train/Adam/AssignAssigntrain/beta1_powertrain/Adam/mul**
_class 
loc:@conv_1/weights/W_conv_1*
_output_shapes
: *
T0*
validate_shape(*
use_locking( 

train/Adam/mul_1Multrain/beta2_power/readtrain/Adam/beta24^train/Adam/update_conv_1/weights/W_conv_1/ApplyAdam3^train/Adam/update_conv_1/biases/b_conv_1/ApplyAdam4^train/Adam/update_conv_2/weights/W_conv_2/ApplyAdam3^train/Adam/update_conv_2/biases/b_conv_2/ApplyAdam4^train/Adam/update_conv_3/weights/W_conv_3/ApplyAdam3^train/Adam/update_conv_3/biases/b_conv_3/ApplyAdam/^train/Adam/update_fc_1/weights/W_fc1/ApplyAdam.^train/Adam/update_fc_1/biases/b_fc1/ApplyAdam/^train/Adam/update_fc_2/weights/W_fc2/ApplyAdam/^train/Adam/update_fc_2/biases/b_fc_2/ApplyAdam**
_class 
loc:@conv_1/weights/W_conv_1*
_output_shapes
: *
T0
И
train/Adam/Assign_1Assigntrain/beta2_powertrain/Adam/mul_1*
use_locking( *
T0**
_class 
loc:@conv_1/weights/W_conv_1*
validate_shape(*
_output_shapes
: 
Р

train/AdamNoOp4^train/Adam/update_conv_1/weights/W_conv_1/ApplyAdam3^train/Adam/update_conv_1/biases/b_conv_1/ApplyAdam4^train/Adam/update_conv_2/weights/W_conv_2/ApplyAdam3^train/Adam/update_conv_2/biases/b_conv_2/ApplyAdam4^train/Adam/update_conv_3/weights/W_conv_3/ApplyAdam3^train/Adam/update_conv_3/biases/b_conv_3/ApplyAdam/^train/Adam/update_fc_1/weights/W_fc1/ApplyAdam.^train/Adam/update_fc_1/biases/b_fc1/ApplyAdam/^train/Adam/update_fc_2/weights/W_fc2/ApplyAdam/^train/Adam/update_fc_2/biases/b_fc_2/ApplyAdam^train/Adam/Assign^train/Adam/Assign_1
[
evaluate/ArgMax/dimensionConst*
value	B :*
_output_shapes
: *
dtype0

evaluate/ArgMaxArgMaxoutputs/Softmaxevaluate/ArgMax/dimension*

Tidx0*
T0*#
_output_shapes
:џџџџџџџџџ
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
:џџџџџџџџџ
i
evaluate/EqualEqualevaluate/ArgMaxevaluate/ArgMax_1*#
_output_shapes
:џџџџџџџџџ*
T0	
b
evaluate/CastCastevaluate/Equal*

SrcT0
*#
_output_shapes
:џџџџџџџџџ*

DstT0
X
evaluate/ConstConst*
valueB: *
_output_shapes
:*
dtype0
r
evaluate/MeanMeanevaluate/Castevaluate/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
`
Merge/MergeSummaryMergeSummarycross_entorpy/cross_entorpy*
N*
_output_shapes
: "ђЉ]$ч8     ВЃb	zфефБBжAJкё
м И 
9
Add
x"T
y"T
z"T"
Ttype:
2	
б
	ApplyAdam
var"T	
m"T	
v"T
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T"
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
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
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
Щ
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
я
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
ю
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

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
О
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
ф
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
2	

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
2	
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

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

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
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring *1.0.12v1.0.0-65-g4763edf-dirtyБ
a
inputs/graphsPlaceholder*)
_output_shapes
:џџџџџџџџџ*
shape: *
dtype0
]
inputs/typePlaceholder*'
_output_shapes
:џџџџџџџџџ*
shape: *
dtype0
v
prepare_tensors/Reshape/shapeConst*%
valueB"џџџџ         *
_output_shapes
:*
dtype0

prepare_tensors/ReshapeReshapeinputs/graphsprepare_tensors/Reshape/shape*
T0*
Tshape0*1
_output_shapes
:џџџџџџџџџ
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
 *ЭЬЬ=*
dtype0*
_output_shapes
: 
Р
/conv_1/weights/truncated_normal/TruncatedNormalTruncatedNormal%conv_1/weights/truncated_normal/shape*&
_output_shapes
:
*
seed2 *
T0*

seed *
dtype0
Д
#conv_1/weights/truncated_normal/mulMul/conv_1/weights/truncated_normal/TruncatedNormal&conv_1/weights/truncated_normal/stddev*
T0*&
_output_shapes
:

Ђ
conv_1/weights/truncated_normalAdd#conv_1/weights/truncated_normal/mul$conv_1/weights/truncated_normal/mean*&
_output_shapes
:
*
T0

conv_1/weights/W_conv_1
VariableV2*&
_output_shapes
:
*
	container *
shape:
*
dtype0*
shared_name 
ш
conv_1/weights/W_conv_1/AssignAssignconv_1/weights/W_conv_1conv_1/weights/truncated_normal*
use_locking(*
T0**
_class 
loc:@conv_1/weights/W_conv_1*
validate_shape(*&
_output_shapes
:


conv_1/weights/W_conv_1/readIdentityconv_1/weights/W_conv_1*
T0**
_class 
loc:@conv_1/weights/W_conv_1*&
_output_shapes
:

`
conv_1/biases/ConstConst*
valueB
*ЭЬЬ=*
_output_shapes
:
*
dtype0

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
Э
conv_1/biases/b_conv_1/AssignAssignconv_1/biases/b_conv_1conv_1/biases/Const*)
_class
loc:@conv_1/biases/b_conv_1*
_output_shapes
:
*
T0*
validate_shape(*
use_locking(

conv_1/biases/b_conv_1/readIdentityconv_1/biases/b_conv_1*)
_class
loc:@conv_1/biases/b_conv_1*
_output_shapes
:
*
T0
ќ
(conv_1/convolution_and_non_linear/Conv2DConv2Dprepare_tensors/Reshapeconv_1/weights/W_conv_1/read*1
_output_shapes
:џџџџџџџџџ
*
T0*
use_cudnn_on_gpu(*
strides
*
data_formatNHWC*
paddingSAME
Џ
%conv_1/convolution_and_non_linear/addAdd(conv_1/convolution_and_non_linear/Conv2Dconv_1/biases/b_conv_1/read*1
_output_shapes
:џџџџџџџџџ
*
T0

&conv_1/convolution_and_non_linear/ReluRelu%conv_1/convolution_and_non_linear/add*1
_output_shapes
:џџџџџџџџџ
*
T0
в
conv_1/pool/MaxPoolMaxPool&conv_1/convolution_and_non_linear/Relu*
ksize
*
T0*
paddingSAME*/
_output_shapes
:џџџџџџџџџ  
*
strides
*
data_formatNHWC
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
 *ЭЬЬ=*
dtype0*
_output_shapes
: 
Р
/conv_2/weights/truncated_normal/TruncatedNormalTruncatedNormal%conv_2/weights/truncated_normal/shape*

seed *
T0*
dtype0*&
_output_shapes
:
*
seed2 
Д
#conv_2/weights/truncated_normal/mulMul/conv_2/weights/truncated_normal/TruncatedNormal&conv_2/weights/truncated_normal/stddev*&
_output_shapes
:
*
T0
Ђ
conv_2/weights/truncated_normalAdd#conv_2/weights/truncated_normal/mul$conv_2/weights/truncated_normal/mean*&
_output_shapes
:
*
T0

conv_2/weights/W_conv_2
VariableV2*
shape:
*
shared_name *
dtype0*&
_output_shapes
:
*
	container 
ш
conv_2/weights/W_conv_2/AssignAssignconv_2/weights/W_conv_2conv_2/weights/truncated_normal**
_class 
loc:@conv_2/weights/W_conv_2*&
_output_shapes
:
*
T0*
validate_shape(*
use_locking(

conv_2/weights/W_conv_2/readIdentityconv_2/weights/W_conv_2*
T0**
_class 
loc:@conv_2/weights/W_conv_2*&
_output_shapes
:

`
conv_2/biases/ConstConst*
valueB*ЭЬЬ=*
dtype0*
_output_shapes
:

conv_2/biases/b_conv_2
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
Э
conv_2/biases/b_conv_2/AssignAssignconv_2/biases/b_conv_2conv_2/biases/Const*
use_locking(*
T0*)
_class
loc:@conv_2/biases/b_conv_2*
validate_shape(*
_output_shapes
:

conv_2/biases/b_conv_2/readIdentityconv_2/biases/b_conv_2*)
_class
loc:@conv_2/biases/b_conv_2*
_output_shapes
:*
T0
і
(conv_2/convolution_and_non_linear/Conv2DConv2Dconv_1/pool/MaxPoolconv_2/weights/W_conv_2/read*
use_cudnn_on_gpu(*
T0*
paddingSAME*/
_output_shapes
:џџџџџџџџџ  *
strides
*
data_formatNHWC
­
%conv_2/convolution_and_non_linear/addAdd(conv_2/convolution_and_non_linear/Conv2Dconv_2/biases/b_conv_2/read*
T0*/
_output_shapes
:џџџџџџџџџ  

&conv_2/convolution_and_non_linear/ReluRelu%conv_2/convolution_and_non_linear/add*
T0*/
_output_shapes
:џџџџџџџџџ  
в
conv_2/pool/MaxPoolMaxPool&conv_2/convolution_and_non_linear/Relu*
ksize
*/
_output_shapes
:џџџџџџџџџ*
T0*
strides
*
data_formatNHWC*
paddingSAME
~
%conv_3/weights/truncated_normal/shapeConst*%
valueB"         (   *
dtype0*
_output_shapes
:
i
$conv_3/weights/truncated_normal/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
k
&conv_3/weights/truncated_normal/stddevConst*
valueB
 *ЭЬЬ=*
dtype0*
_output_shapes
: 
Р
/conv_3/weights/truncated_normal/TruncatedNormalTruncatedNormal%conv_3/weights/truncated_normal/shape*&
_output_shapes
:(*
seed2 *
T0*

seed *
dtype0
Д
#conv_3/weights/truncated_normal/mulMul/conv_3/weights/truncated_normal/TruncatedNormal&conv_3/weights/truncated_normal/stddev*&
_output_shapes
:(*
T0
Ђ
conv_3/weights/truncated_normalAdd#conv_3/weights/truncated_normal/mul$conv_3/weights/truncated_normal/mean*
T0*&
_output_shapes
:(

conv_3/weights/W_conv_3
VariableV2*&
_output_shapes
:(*
	container *
shape:(*
dtype0*
shared_name 
ш
conv_3/weights/W_conv_3/AssignAssignconv_3/weights/W_conv_3conv_3/weights/truncated_normal**
_class 
loc:@conv_3/weights/W_conv_3*&
_output_shapes
:(*
T0*
validate_shape(*
use_locking(

conv_3/weights/W_conv_3/readIdentityconv_3/weights/W_conv_3**
_class 
loc:@conv_3/weights/W_conv_3*&
_output_shapes
:(*
T0
`
conv_3/biases/ConstConst*
valueB(*ЭЬЬ=*
_output_shapes
:(*
dtype0

conv_3/biases/b_conv_3
VariableV2*
shape:(*
shared_name *
dtype0*
_output_shapes
:(*
	container 
Э
conv_3/biases/b_conv_3/AssignAssignconv_3/biases/b_conv_3conv_3/biases/Const*)
_class
loc:@conv_3/biases/b_conv_3*
_output_shapes
:(*
T0*
validate_shape(*
use_locking(

conv_3/biases/b_conv_3/readIdentityconv_3/biases/b_conv_3*
T0*)
_class
loc:@conv_3/biases/b_conv_3*
_output_shapes
:(
і
(conv_3/convolution_and_non_linear/Conv2DConv2Dconv_2/pool/MaxPoolconv_3/weights/W_conv_3/read*
use_cudnn_on_gpu(*
T0*
paddingSAME*/
_output_shapes
:џџџџџџџџџ(*
strides
*
data_formatNHWC
­
%conv_3/convolution_and_non_linear/addAdd(conv_3/convolution_and_non_linear/Conv2Dconv_3/biases/b_conv_3/read*/
_output_shapes
:џџџџџџџџџ(*
T0

&conv_3/convolution_and_non_linear/ReluRelu%conv_3/convolution_and_non_linear/add*
T0*/
_output_shapes
:џџџџџџџџџ(
в
conv_3/pool/MaxPoolMaxPool&conv_3/convolution_and_non_linear/Relu*
paddingSAME*
T0*
strides
*
data_formatNHWC*/
_output_shapes
:џџџџџџџџџ(*
ksize

f
flatten/Reshape/shapeConst*
valueB"џџџџ 
  *
_output_shapes
:*
dtype0

flatten/ReshapeReshapeconv_3/pool/MaxPoolflatten/Reshape/shape*
Tshape0*(
_output_shapes
:џџџџџџџџџ*
T0
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
 *ЭЬЬ=*
_output_shapes
: *
dtype0
Е
-fc_1/weights/truncated_normal/TruncatedNormalTruncatedNormal#fc_1/weights/truncated_normal/shape*
_output_shapes
:	d*
seed2 *
T0*

seed *
dtype0
Ї
!fc_1/weights/truncated_normal/mulMul-fc_1/weights/truncated_normal/TruncatedNormal$fc_1/weights/truncated_normal/stddev*
T0*
_output_shapes
:	d

fc_1/weights/truncated_normalAdd!fc_1/weights/truncated_normal/mul"fc_1/weights/truncated_normal/mean*
_output_shapes
:	d*
T0

fc_1/weights/W_fc1
VariableV2*
shape:	d*
shared_name *
dtype0*
_output_shapes
:	d*
	container 
а
fc_1/weights/W_fc1/AssignAssignfc_1/weights/W_fc1fc_1/weights/truncated_normal*
use_locking(*
T0*%
_class
loc:@fc_1/weights/W_fc1*
validate_shape(*
_output_shapes
:	d

fc_1/weights/W_fc1/readIdentityfc_1/weights/W_fc1*
T0*%
_class
loc:@fc_1/weights/W_fc1*
_output_shapes
:	d
^
fc_1/biases/ConstConst*
valueBd*ЭЬЬ=*
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
М
fc_1/biases/b_fc1/AssignAssignfc_1/biases/b_fc1fc_1/biases/Const*$
_class
loc:@fc_1/biases/b_fc1*
_output_shapes
:d*
T0*
validate_shape(*
use_locking(

fc_1/biases/b_fc1/readIdentityfc_1/biases/b_fc1*
T0*$
_class
loc:@fc_1/biases/b_fc1*
_output_shapes
:d
­
!fc_1/matmul_and_non_linear/MatMulMatMulflatten/Reshapefc_1/weights/W_fc1/read*
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџd*
transpose_a( 

fc_1/matmul_and_non_linear/addAdd!fc_1/matmul_and_non_linear/MatMulfc_1/biases/b_fc1/read*'
_output_shapes
:џџџџџџџџџd*
T0
y
fc_1/matmul_and_non_linear/ReluRelufc_1/matmul_and_non_linear/add*
T0*'
_output_shapes
:џџџџџџџџџd
t
#fc_2/weights/truncated_normal/shapeConst*
valueB"d      *
_output_shapes
:*
dtype0
g
"fc_2/weights/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
i
$fc_2/weights/truncated_normal/stddevConst*
valueB
 *ЭЬЬ=*
dtype0*
_output_shapes
: 
Д
-fc_2/weights/truncated_normal/TruncatedNormalTruncatedNormal#fc_2/weights/truncated_normal/shape*

seed *
T0*
dtype0*
_output_shapes

:d*
seed2 
І
!fc_2/weights/truncated_normal/mulMul-fc_2/weights/truncated_normal/TruncatedNormal$fc_2/weights/truncated_normal/stddev*
_output_shapes

:d*
T0

fc_2/weights/truncated_normalAdd!fc_2/weights/truncated_normal/mul"fc_2/weights/truncated_normal/mean*
_output_shapes

:d*
T0

fc_2/weights/W_fc2
VariableV2*
_output_shapes

:d*
	container *
shape
:d*
dtype0*
shared_name 
Я
fc_2/weights/W_fc2/AssignAssignfc_2/weights/W_fc2fc_2/weights/truncated_normal*
use_locking(*
T0*%
_class
loc:@fc_2/weights/W_fc2*
validate_shape(*
_output_shapes

:d

fc_2/weights/W_fc2/readIdentityfc_2/weights/W_fc2*%
_class
loc:@fc_2/weights/W_fc2*
_output_shapes

:d*
T0
^
fc_2/biases/ConstConst*
valueB*ЭЬЬ=*
_output_shapes
:*
dtype0
~
fc_2/biases/b_fc_2
VariableV2*
_output_shapes
:*
	container *
shape:*
dtype0*
shared_name 
П
fc_2/biases/b_fc_2/AssignAssignfc_2/biases/b_fc_2fc_2/biases/Const*%
_class
loc:@fc_2/biases/b_fc_2*
_output_shapes
:*
T0*
validate_shape(*
use_locking(

fc_2/biases/b_fc_2/readIdentityfc_2/biases/b_fc_2*%
_class
loc:@fc_2/biases/b_fc_2*
_output_shapes
:*
T0
Њ
outputs/MatMulMatMulfc_1/matmul_and_non_linear/Relufc_2/weights/W_fc2/read*
transpose_b( *
T0*'
_output_shapes
:џџџџџџџџџ*
transpose_a( 
m
outputs/addAddoutputs/MatMulfc_2/biases/b_fc_2/read*
T0*'
_output_shapes
:џџџџџџџџџ
Y
outputs/SoftmaxSoftmaxoutputs/add*'
_output_shapes
:џџџџџџџџџ*
T0
[
cross_entorpy/LogLogoutputs/Softmax*
T0*'
_output_shapes
:џџџџџџџџџ
j
cross_entorpy/mulMulinputs/typecross_entorpy/Log*'
_output_shapes
:џџџџџџџџџ*
T0
m
#cross_entorpy/Sum/reduction_indicesConst*
valueB:*
_output_shapes
:*
dtype0

cross_entorpy/SumSumcross_entorpy/mul#cross_entorpy/Sum/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:џџџџџџџџџ
Y
cross_entorpy/NegNegcross_entorpy/Sum*#
_output_shapes
:џџџџџџџџџ*
T0
]
cross_entorpy/ConstConst*
valueB: *
_output_shapes
:*
dtype0

cross_entorpy/MeanMeancross_entorpy/Negcross_entorpy/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
|
 cross_entorpy/cross_entorpy/tagsConst*,
value#B! Bcross_entorpy/cross_entorpy*
dtype0*
_output_shapes
: 

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
 *  ?*
dtype0*
_output_shapes
: 
k
train/gradients/FillFilltrain/gradients/Shapetrain/gradients/Const*
T0*
_output_shapes
: 

5train/gradients/cross_entorpy/Mean_grad/Reshape/shapeConst*
valueB:*
_output_shapes
:*
dtype0
К
/train/gradients/cross_entorpy/Mean_grad/ReshapeReshapetrain/gradients/Fill5train/gradients/cross_entorpy/Mean_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
~
-train/gradients/cross_entorpy/Mean_grad/ShapeShapecross_entorpy/Neg*
T0*
out_type0*
_output_shapes
:
д
,train/gradients/cross_entorpy/Mean_grad/TileTile/train/gradients/cross_entorpy/Mean_grad/Reshape-train/gradients/cross_entorpy/Mean_grad/Shape*#
_output_shapes
:џџџџџџџџџ*
T0*

Tmultiples0

/train/gradients/cross_entorpy/Mean_grad/Shape_1Shapecross_entorpy/Neg*
out_type0*
_output_shapes
:*
T0
r
/train/gradients/cross_entorpy/Mean_grad/Shape_2Const*
valueB *
_output_shapes
: *
dtype0
w
-train/gradients/cross_entorpy/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
в
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
ж
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
О
/train/gradients/cross_entorpy/Mean_grad/MaximumMaximum.train/gradients/cross_entorpy/Mean_grad/Prod_11train/gradients/cross_entorpy/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
М
0train/gradients/cross_entorpy/Mean_grad/floordivFloorDiv,train/gradients/cross_entorpy/Mean_grad/Prod/train/gradients/cross_entorpy/Mean_grad/Maximum*
T0*
_output_shapes
: 

,train/gradients/cross_entorpy/Mean_grad/CastCast0train/gradients/cross_entorpy/Mean_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0
Ф
/train/gradients/cross_entorpy/Mean_grad/truedivRealDiv,train/gradients/cross_entorpy/Mean_grad/Tile,train/gradients/cross_entorpy/Mean_grad/Cast*#
_output_shapes
:џџџџџџџџџ*
T0

*train/gradients/cross_entorpy/Neg_grad/NegNeg/train/gradients/cross_entorpy/Mean_grad/truediv*#
_output_shapes
:џџџџџџџџџ*
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
Ј
*train/gradients/cross_entorpy/Sum_grad/addAdd#cross_entorpy/Sum/reduction_indices+train/gradients/cross_entorpy/Sum_grad/Size*
_output_shapes
:*
T0
Д
*train/gradients/cross_entorpy/Sum_grad/modFloorMod*train/gradients/cross_entorpy/Sum_grad/add+train/gradients/cross_entorpy/Sum_grad/Size*
T0*
_output_shapes
:
x
.train/gradients/cross_entorpy/Sum_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
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
ђ
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
Л
+train/gradients/cross_entorpy/Sum_grad/FillFill.train/gradients/cross_entorpy/Sum_grad/Shape_11train/gradients/cross_entorpy/Sum_grad/Fill/value*
T0*
_output_shapes
:
Б
4train/gradients/cross_entorpy/Sum_grad/DynamicStitchDynamicStitch,train/gradients/cross_entorpy/Sum_grad/range*train/gradients/cross_entorpy/Sum_grad/mod,train/gradients/cross_entorpy/Sum_grad/Shape+train/gradients/cross_entorpy/Sum_grad/Fill*
T0*
N*#
_output_shapes
:џџџџџџџџџ
r
0train/gradients/cross_entorpy/Sum_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
Я
.train/gradients/cross_entorpy/Sum_grad/MaximumMaximum4train/gradients/cross_entorpy/Sum_grad/DynamicStitch0train/gradients/cross_entorpy/Sum_grad/Maximum/y*#
_output_shapes
:џџџџџџџџџ*
T0
О
/train/gradients/cross_entorpy/Sum_grad/floordivFloorDiv,train/gradients/cross_entorpy/Sum_grad/Shape.train/gradients/cross_entorpy/Sum_grad/Maximum*
_output_shapes
:*
T0
Ь
.train/gradients/cross_entorpy/Sum_grad/ReshapeReshape*train/gradients/cross_entorpy/Neg_grad/Neg4train/gradients/cross_entorpy/Sum_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
и
+train/gradients/cross_entorpy/Sum_grad/TileTile.train/gradients/cross_entorpy/Sum_grad/Reshape/train/gradients/cross_entorpy/Sum_grad/floordiv*

Tmultiples0*
T0*'
_output_shapes
:џџџџџџџџџ
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
№
<train/gradients/cross_entorpy/mul_grad/BroadcastGradientArgsBroadcastGradientArgs,train/gradients/cross_entorpy/mul_grad/Shape.train/gradients/cross_entorpy/mul_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ѓ
*train/gradients/cross_entorpy/mul_grad/mulMul+train/gradients/cross_entorpy/Sum_grad/Tilecross_entorpy/Log*'
_output_shapes
:џџџџџџџџџ*
T0
л
*train/gradients/cross_entorpy/mul_grad/SumSum*train/gradients/cross_entorpy/mul_grad/mul<train/gradients/cross_entorpy/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
г
.train/gradients/cross_entorpy/mul_grad/ReshapeReshape*train/gradients/cross_entorpy/mul_grad/Sum,train/gradients/cross_entorpy/mul_grad/Shape*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0

,train/gradients/cross_entorpy/mul_grad/mul_1Mulinputs/type+train/gradients/cross_entorpy/Sum_grad/Tile*'
_output_shapes
:џџџџџџџџџ*
T0
с
,train/gradients/cross_entorpy/mul_grad/Sum_1Sum,train/gradients/cross_entorpy/mul_grad/mul_1>train/gradients/cross_entorpy/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
й
0train/gradients/cross_entorpy/mul_grad/Reshape_1Reshape,train/gradients/cross_entorpy/mul_grad/Sum_1.train/gradients/cross_entorpy/mul_grad/Shape_1*
Tshape0*'
_output_shapes
:џџџџџџџџџ*
T0
Ѓ
7train/gradients/cross_entorpy/mul_grad/tuple/group_depsNoOp/^train/gradients/cross_entorpy/mul_grad/Reshape1^train/gradients/cross_entorpy/mul_grad/Reshape_1
Њ
?train/gradients/cross_entorpy/mul_grad/tuple/control_dependencyIdentity.train/gradients/cross_entorpy/mul_grad/Reshape8^train/gradients/cross_entorpy/mul_grad/tuple/group_deps*A
_class7
53loc:@train/gradients/cross_entorpy/mul_grad/Reshape*'
_output_shapes
:џџџџџџџџџ*
T0
А
Atrain/gradients/cross_entorpy/mul_grad/tuple/control_dependency_1Identity0train/gradients/cross_entorpy/mul_grad/Reshape_18^train/gradients/cross_entorpy/mul_grad/tuple/group_deps*C
_class9
75loc:@train/gradients/cross_entorpy/mul_grad/Reshape_1*'
_output_shapes
:џџџџџџџџџ*
T0
Ц
1train/gradients/cross_entorpy/Log_grad/Reciprocal
Reciprocaloutputs/SoftmaxB^train/gradients/cross_entorpy/mul_grad/tuple/control_dependency_1*'
_output_shapes
:џџџџџџџџџ*
T0
й
*train/gradients/cross_entorpy/Log_grad/mulMulAtrain/gradients/cross_entorpy/mul_grad/tuple/control_dependency_11train/gradients/cross_entorpy/Log_grad/Reciprocal*
T0*'
_output_shapes
:џџџџџџџџџ

(train/gradients/outputs/Softmax_grad/mulMul*train/gradients/cross_entorpy/Log_grad/muloutputs/Softmax*
T0*'
_output_shapes
:џџџџџџџџџ

:train/gradients/outputs/Softmax_grad/Sum/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
р
(train/gradients/outputs/Softmax_grad/SumSum(train/gradients/outputs/Softmax_grad/mul:train/gradients/outputs/Softmax_grad/Sum/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:џџџџџџџџџ

2train/gradients/outputs/Softmax_grad/Reshape/shapeConst*
valueB"џџџџ   *
dtype0*
_output_shapes
:
е
,train/gradients/outputs/Softmax_grad/ReshapeReshape(train/gradients/outputs/Softmax_grad/Sum2train/gradients/outputs/Softmax_grad/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
Л
(train/gradients/outputs/Softmax_grad/subSub*train/gradients/cross_entorpy/Log_grad/mul,train/gradients/outputs/Softmax_grad/Reshape*
T0*'
_output_shapes
:џџџџџџџџџ

*train/gradients/outputs/Softmax_grad/mul_1Mul(train/gradients/outputs/Softmax_grad/suboutputs/Softmax*'
_output_shapes
:џџџџџџџџџ*
T0
t
&train/gradients/outputs/add_grad/ShapeShapeoutputs/MatMul*
T0*
out_type0*
_output_shapes
:
r
(train/gradients/outputs/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
о
6train/gradients/outputs/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/outputs/add_grad/Shape(train/gradients/outputs/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Я
$train/gradients/outputs/add_grad/SumSum*train/gradients/outputs/Softmax_grad/mul_16train/gradients/outputs/add_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
С
(train/gradients/outputs/add_grad/ReshapeReshape$train/gradients/outputs/add_grad/Sum&train/gradients/outputs/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ
г
&train/gradients/outputs/add_grad/Sum_1Sum*train/gradients/outputs/Softmax_grad/mul_18train/gradients/outputs/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
К
*train/gradients/outputs/add_grad/Reshape_1Reshape&train/gradients/outputs/add_grad/Sum_1(train/gradients/outputs/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:

1train/gradients/outputs/add_grad/tuple/group_depsNoOp)^train/gradients/outputs/add_grad/Reshape+^train/gradients/outputs/add_grad/Reshape_1

9train/gradients/outputs/add_grad/tuple/control_dependencyIdentity(train/gradients/outputs/add_grad/Reshape2^train/gradients/outputs/add_grad/tuple/group_deps*;
_class1
/-loc:@train/gradients/outputs/add_grad/Reshape*'
_output_shapes
:џџџџџџџџџ*
T0

;train/gradients/outputs/add_grad/tuple/control_dependency_1Identity*train/gradients/outputs/add_grad/Reshape_12^train/gradients/outputs/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/outputs/add_grad/Reshape_1*
_output_shapes
:
р
*train/gradients/outputs/MatMul_grad/MatMulMatMul9train/gradients/outputs/add_grad/tuple/control_dependencyfc_2/weights/W_fc2/read*
transpose_b(*'
_output_shapes
:џџџџџџџџџd*
transpose_a( *
T0
с
,train/gradients/outputs/MatMul_grad/MatMul_1MatMulfc_1/matmul_and_non_linear/Relu9train/gradients/outputs/add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:d*
transpose_a(

4train/gradients/outputs/MatMul_grad/tuple/group_depsNoOp+^train/gradients/outputs/MatMul_grad/MatMul-^train/gradients/outputs/MatMul_grad/MatMul_1

<train/gradients/outputs/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/outputs/MatMul_grad/MatMul5^train/gradients/outputs/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/outputs/MatMul_grad/MatMul*'
_output_shapes
:џџџџџџџџџd

>train/gradients/outputs/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/outputs/MatMul_grad/MatMul_15^train/gradients/outputs/MatMul_grad/tuple/group_deps*?
_class5
31loc:@train/gradients/outputs/MatMul_grad/MatMul_1*
_output_shapes

:d*
T0
к
=train/gradients/fc_1/matmul_and_non_linear/Relu_grad/ReluGradReluGrad<train/gradients/outputs/MatMul_grad/tuple/control_dependencyfc_1/matmul_and_non_linear/Relu*
T0*'
_output_shapes
:џџџџџџџџџd

9train/gradients/fc_1/matmul_and_non_linear/add_grad/ShapeShape!fc_1/matmul_and_non_linear/MatMul*
out_type0*
_output_shapes
:*
T0

;train/gradients/fc_1/matmul_and_non_linear/add_grad/Shape_1Const*
valueB:d*
dtype0*
_output_shapes
:

Itrain/gradients/fc_1/matmul_and_non_linear/add_grad/BroadcastGradientArgsBroadcastGradientArgs9train/gradients/fc_1/matmul_and_non_linear/add_grad/Shape;train/gradients/fc_1/matmul_and_non_linear/add_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

7train/gradients/fc_1/matmul_and_non_linear/add_grad/SumSum=train/gradients/fc_1/matmul_and_non_linear/Relu_grad/ReluGradItrain/gradients/fc_1/matmul_and_non_linear/add_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
њ
;train/gradients/fc_1/matmul_and_non_linear/add_grad/ReshapeReshape7train/gradients/fc_1/matmul_and_non_linear/add_grad/Sum9train/gradients/fc_1/matmul_and_non_linear/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџd

9train/gradients/fc_1/matmul_and_non_linear/add_grad/Sum_1Sum=train/gradients/fc_1/matmul_and_non_linear/Relu_grad/ReluGradKtrain/gradients/fc_1/matmul_and_non_linear/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
ѓ
=train/gradients/fc_1/matmul_and_non_linear/add_grad/Reshape_1Reshape9train/gradients/fc_1/matmul_and_non_linear/add_grad/Sum_1;train/gradients/fc_1/matmul_and_non_linear/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:d
Ъ
Dtrain/gradients/fc_1/matmul_and_non_linear/add_grad/tuple/group_depsNoOp<^train/gradients/fc_1/matmul_and_non_linear/add_grad/Reshape>^train/gradients/fc_1/matmul_and_non_linear/add_grad/Reshape_1
о
Ltrain/gradients/fc_1/matmul_and_non_linear/add_grad/tuple/control_dependencyIdentity;train/gradients/fc_1/matmul_and_non_linear/add_grad/ReshapeE^train/gradients/fc_1/matmul_and_non_linear/add_grad/tuple/group_deps*N
_classD
B@loc:@train/gradients/fc_1/matmul_and_non_linear/add_grad/Reshape*'
_output_shapes
:џџџџџџџџџd*
T0
з
Ntrain/gradients/fc_1/matmul_and_non_linear/add_grad/tuple/control_dependency_1Identity=train/gradients/fc_1/matmul_and_non_linear/add_grad/Reshape_1E^train/gradients/fc_1/matmul_and_non_linear/add_grad/tuple/group_deps*P
_classF
DBloc:@train/gradients/fc_1/matmul_and_non_linear/add_grad/Reshape_1*
_output_shapes
:d*
T0

=train/gradients/fc_1/matmul_and_non_linear/MatMul_grad/MatMulMatMulLtrain/gradients/fc_1/matmul_and_non_linear/add_grad/tuple/control_dependencyfc_1/weights/W_fc1/read*
transpose_b(*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( 
ј
?train/gradients/fc_1/matmul_and_non_linear/MatMul_grad/MatMul_1MatMulflatten/ReshapeLtrain/gradients/fc_1/matmul_and_non_linear/add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes
:	d*
transpose_a(
б
Gtrain/gradients/fc_1/matmul_and_non_linear/MatMul_grad/tuple/group_depsNoOp>^train/gradients/fc_1/matmul_and_non_linear/MatMul_grad/MatMul@^train/gradients/fc_1/matmul_and_non_linear/MatMul_grad/MatMul_1
щ
Otrain/gradients/fc_1/matmul_and_non_linear/MatMul_grad/tuple/control_dependencyIdentity=train/gradients/fc_1/matmul_and_non_linear/MatMul_grad/MatMulH^train/gradients/fc_1/matmul_and_non_linear/MatMul_grad/tuple/group_deps*
T0*P
_classF
DBloc:@train/gradients/fc_1/matmul_and_non_linear/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ
ц
Qtrain/gradients/fc_1/matmul_and_non_linear/MatMul_grad/tuple/control_dependency_1Identity?train/gradients/fc_1/matmul_and_non_linear/MatMul_grad/MatMul_1H^train/gradients/fc_1/matmul_and_non_linear/MatMul_grad/tuple/group_deps*
T0*R
_classH
FDloc:@train/gradients/fc_1/matmul_and_non_linear/MatMul_grad/MatMul_1*
_output_shapes
:	d
}
*train/gradients/flatten/Reshape_grad/ShapeShapeconv_3/pool/MaxPool*
T0*
out_type0*
_output_shapes
:
ќ
,train/gradients/flatten/Reshape_grad/ReshapeReshapeOtrain/gradients/fc_1/matmul_and_non_linear/MatMul_grad/tuple/control_dependency*train/gradients/flatten/Reshape_grad/Shape*
Tshape0*/
_output_shapes
:џџџџџџџџџ(*
T0
К
4train/gradients/conv_3/pool/MaxPool_grad/MaxPoolGradMaxPoolGrad&conv_3/convolution_and_non_linear/Reluconv_3/pool/MaxPool,train/gradients/flatten/Reshape_grad/Reshape*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ(
ш
Dtrain/gradients/conv_3/convolution_and_non_linear/Relu_grad/ReluGradReluGrad4train/gradients/conv_3/pool/MaxPool_grad/MaxPoolGrad&conv_3/convolution_and_non_linear/Relu*
T0*/
_output_shapes
:џџџџџџџџџ(
Ј
@train/gradients/conv_3/convolution_and_non_linear/add_grad/ShapeShape(conv_3/convolution_and_non_linear/Conv2D*
T0*
out_type0*
_output_shapes
:

Btrain/gradients/conv_3/convolution_and_non_linear/add_grad/Shape_1Const*
valueB:(*
dtype0*
_output_shapes
:
Ќ
Ptrain/gradients/conv_3/convolution_and_non_linear/add_grad/BroadcastGradientArgsBroadcastGradientArgs@train/gradients/conv_3/convolution_and_non_linear/add_grad/ShapeBtrain/gradients/conv_3/convolution_and_non_linear/add_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

>train/gradients/conv_3/convolution_and_non_linear/add_grad/SumSumDtrain/gradients/conv_3/convolution_and_non_linear/Relu_grad/ReluGradPtrain/gradients/conv_3/convolution_and_non_linear/add_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0

Btrain/gradients/conv_3/convolution_and_non_linear/add_grad/ReshapeReshape>train/gradients/conv_3/convolution_and_non_linear/add_grad/Sum@train/gradients/conv_3/convolution_and_non_linear/add_grad/Shape*
T0*
Tshape0*/
_output_shapes
:џџџџџџџџџ(
Ё
@train/gradients/conv_3/convolution_and_non_linear/add_grad/Sum_1SumDtrain/gradients/conv_3/convolution_and_non_linear/Relu_grad/ReluGradRtrain/gradients/conv_3/convolution_and_non_linear/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0

Dtrain/gradients/conv_3/convolution_and_non_linear/add_grad/Reshape_1Reshape@train/gradients/conv_3/convolution_and_non_linear/add_grad/Sum_1Btrain/gradients/conv_3/convolution_and_non_linear/add_grad/Shape_1*
Tshape0*
_output_shapes
:(*
T0
п
Ktrain/gradients/conv_3/convolution_and_non_linear/add_grad/tuple/group_depsNoOpC^train/gradients/conv_3/convolution_and_non_linear/add_grad/ReshapeE^train/gradients/conv_3/convolution_and_non_linear/add_grad/Reshape_1

Strain/gradients/conv_3/convolution_and_non_linear/add_grad/tuple/control_dependencyIdentityBtrain/gradients/conv_3/convolution_and_non_linear/add_grad/ReshapeL^train/gradients/conv_3/convolution_and_non_linear/add_grad/tuple/group_deps*U
_classK
IGloc:@train/gradients/conv_3/convolution_and_non_linear/add_grad/Reshape*/
_output_shapes
:џџџџџџџџџ(*
T0
ѓ
Utrain/gradients/conv_3/convolution_and_non_linear/add_grad/tuple/control_dependency_1IdentityDtrain/gradients/conv_3/convolution_and_non_linear/add_grad/Reshape_1L^train/gradients/conv_3/convolution_and_non_linear/add_grad/tuple/group_deps*W
_classM
KIloc:@train/gradients/conv_3/convolution_and_non_linear/add_grad/Reshape_1*
_output_shapes
:(*
T0

Ctrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/ShapeShapeconv_2/pool/MaxPool*
out_type0*
_output_shapes
:*
T0
Ь
Qtrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputCtrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/Shapeconv_3/weights/W_conv_3/readStrain/gradients/conv_3/convolution_and_non_linear/add_grad/tuple/control_dependency*
paddingSAME*
T0*
data_formatNHWC*
strides
*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
use_cudnn_on_gpu(

Etrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/Shape_1Const*%
valueB"         (   *
dtype0*
_output_shapes
:
Ѓ
Rtrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterconv_2/pool/MaxPoolEtrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/Shape_1Strain/gradients/conv_3/convolution_and_non_linear/add_grad/tuple/control_dependency*&
_output_shapes
:(*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingSAME
џ
Ntrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/tuple/group_depsNoOpR^train/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInputS^train/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilter
І
Vtrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/tuple/control_dependencyIdentityQtrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInputO^train/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@train/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:џџџџџџџџџ
Ё
Xtrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/tuple/control_dependency_1IdentityRtrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilterO^train/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/tuple/group_deps*
T0*e
_class[
YWloc:@train/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:(
ф
4train/gradients/conv_2/pool/MaxPool_grad/MaxPoolGradMaxPoolGrad&conv_2/convolution_and_non_linear/Reluconv_2/pool/MaxPoolVtrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/tuple/control_dependency*
ksize
*
T0*
paddingSAME*/
_output_shapes
:џџџџџџџџџ  *
data_formatNHWC*
strides

ш
Dtrain/gradients/conv_2/convolution_and_non_linear/Relu_grad/ReluGradReluGrad4train/gradients/conv_2/pool/MaxPool_grad/MaxPoolGrad&conv_2/convolution_and_non_linear/Relu*
T0*/
_output_shapes
:џџџџџџџџџ  
Ј
@train/gradients/conv_2/convolution_and_non_linear/add_grad/ShapeShape(conv_2/convolution_and_non_linear/Conv2D*
T0*
out_type0*
_output_shapes
:

Btrain/gradients/conv_2/convolution_and_non_linear/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
Ќ
Ptrain/gradients/conv_2/convolution_and_non_linear/add_grad/BroadcastGradientArgsBroadcastGradientArgs@train/gradients/conv_2/convolution_and_non_linear/add_grad/ShapeBtrain/gradients/conv_2/convolution_and_non_linear/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

>train/gradients/conv_2/convolution_and_non_linear/add_grad/SumSumDtrain/gradients/conv_2/convolution_and_non_linear/Relu_grad/ReluGradPtrain/gradients/conv_2/convolution_and_non_linear/add_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0

Btrain/gradients/conv_2/convolution_and_non_linear/add_grad/ReshapeReshape>train/gradients/conv_2/convolution_and_non_linear/add_grad/Sum@train/gradients/conv_2/convolution_and_non_linear/add_grad/Shape*
T0*
Tshape0*/
_output_shapes
:џџџџџџџџџ  
Ё
@train/gradients/conv_2/convolution_and_non_linear/add_grad/Sum_1SumDtrain/gradients/conv_2/convolution_and_non_linear/Relu_grad/ReluGradRtrain/gradients/conv_2/convolution_and_non_linear/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:

Dtrain/gradients/conv_2/convolution_and_non_linear/add_grad/Reshape_1Reshape@train/gradients/conv_2/convolution_and_non_linear/add_grad/Sum_1Btrain/gradients/conv_2/convolution_and_non_linear/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
п
Ktrain/gradients/conv_2/convolution_and_non_linear/add_grad/tuple/group_depsNoOpC^train/gradients/conv_2/convolution_and_non_linear/add_grad/ReshapeE^train/gradients/conv_2/convolution_and_non_linear/add_grad/Reshape_1

Strain/gradients/conv_2/convolution_and_non_linear/add_grad/tuple/control_dependencyIdentityBtrain/gradients/conv_2/convolution_and_non_linear/add_grad/ReshapeL^train/gradients/conv_2/convolution_and_non_linear/add_grad/tuple/group_deps*
T0*U
_classK
IGloc:@train/gradients/conv_2/convolution_and_non_linear/add_grad/Reshape*/
_output_shapes
:џџџџџџџџџ  
ѓ
Utrain/gradients/conv_2/convolution_and_non_linear/add_grad/tuple/control_dependency_1IdentityDtrain/gradients/conv_2/convolution_and_non_linear/add_grad/Reshape_1L^train/gradients/conv_2/convolution_and_non_linear/add_grad/tuple/group_deps*W
_classM
KIloc:@train/gradients/conv_2/convolution_and_non_linear/add_grad/Reshape_1*
_output_shapes
:*
T0

Ctrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/ShapeShapeconv_1/pool/MaxPool*
T0*
out_type0*
_output_shapes
:
Ь
Qtrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputCtrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/Shapeconv_2/weights/W_conv_2/readStrain/gradients/conv_2/convolution_and_non_linear/add_grad/tuple/control_dependency*
paddingSAME*
T0*
data_formatNHWC*
strides
*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
use_cudnn_on_gpu(

Etrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/Shape_1Const*%
valueB"      
      *
dtype0*
_output_shapes
:
Ѓ
Rtrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterconv_1/pool/MaxPoolEtrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/Shape_1Strain/gradients/conv_2/convolution_and_non_linear/add_grad/tuple/control_dependency*&
_output_shapes
:
*
T0*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
paddingSAME
џ
Ntrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/tuple/group_depsNoOpR^train/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInputS^train/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilter
І
Vtrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/tuple/control_dependencyIdentityQtrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInputO^train/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/tuple/group_deps*d
_classZ
XVloc:@train/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:џџџџџџџџџ  
*
T0
Ё
Xtrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/tuple/control_dependency_1IdentityRtrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilterO^train/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/tuple/group_deps*
T0*e
_class[
YWloc:@train/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:

ц
4train/gradients/conv_1/pool/MaxPool_grad/MaxPoolGradMaxPoolGrad&conv_1/convolution_and_non_linear/Reluconv_1/pool/MaxPoolVtrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/tuple/control_dependency*
paddingSAME*
T0*
data_formatNHWC*
strides
*1
_output_shapes
:џџџџџџџџџ
*
ksize

ъ
Dtrain/gradients/conv_1/convolution_and_non_linear/Relu_grad/ReluGradReluGrad4train/gradients/conv_1/pool/MaxPool_grad/MaxPoolGrad&conv_1/convolution_and_non_linear/Relu*
T0*1
_output_shapes
:џџџџџџџџџ

Ј
@train/gradients/conv_1/convolution_and_non_linear/add_grad/ShapeShape(conv_1/convolution_and_non_linear/Conv2D*
out_type0*
_output_shapes
:*
T0

Btrain/gradients/conv_1/convolution_and_non_linear/add_grad/Shape_1Const*
valueB:
*
_output_shapes
:*
dtype0
Ќ
Ptrain/gradients/conv_1/convolution_and_non_linear/add_grad/BroadcastGradientArgsBroadcastGradientArgs@train/gradients/conv_1/convolution_and_non_linear/add_grad/ShapeBtrain/gradients/conv_1/convolution_and_non_linear/add_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

>train/gradients/conv_1/convolution_and_non_linear/add_grad/SumSumDtrain/gradients/conv_1/convolution_and_non_linear/Relu_grad/ReluGradPtrain/gradients/conv_1/convolution_and_non_linear/add_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0

Btrain/gradients/conv_1/convolution_and_non_linear/add_grad/ReshapeReshape>train/gradients/conv_1/convolution_and_non_linear/add_grad/Sum@train/gradients/conv_1/convolution_and_non_linear/add_grad/Shape*
T0*
Tshape0*1
_output_shapes
:џџџџџџџџџ

Ё
@train/gradients/conv_1/convolution_and_non_linear/add_grad/Sum_1SumDtrain/gradients/conv_1/convolution_and_non_linear/Relu_grad/ReluGradRtrain/gradients/conv_1/convolution_and_non_linear/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0

Dtrain/gradients/conv_1/convolution_and_non_linear/add_grad/Reshape_1Reshape@train/gradients/conv_1/convolution_and_non_linear/add_grad/Sum_1Btrain/gradients/conv_1/convolution_and_non_linear/add_grad/Shape_1*
Tshape0*
_output_shapes
:
*
T0
п
Ktrain/gradients/conv_1/convolution_and_non_linear/add_grad/tuple/group_depsNoOpC^train/gradients/conv_1/convolution_and_non_linear/add_grad/ReshapeE^train/gradients/conv_1/convolution_and_non_linear/add_grad/Reshape_1

Strain/gradients/conv_1/convolution_and_non_linear/add_grad/tuple/control_dependencyIdentityBtrain/gradients/conv_1/convolution_and_non_linear/add_grad/ReshapeL^train/gradients/conv_1/convolution_and_non_linear/add_grad/tuple/group_deps*
T0*U
_classK
IGloc:@train/gradients/conv_1/convolution_and_non_linear/add_grad/Reshape*1
_output_shapes
:џџџџџџџџџ

ѓ
Utrain/gradients/conv_1/convolution_and_non_linear/add_grad/tuple/control_dependency_1IdentityDtrain/gradients/conv_1/convolution_and_non_linear/add_grad/Reshape_1L^train/gradients/conv_1/convolution_and_non_linear/add_grad/tuple/group_deps*
T0*W
_classM
KIloc:@train/gradients/conv_1/convolution_and_non_linear/add_grad/Reshape_1*
_output_shapes
:


Ctrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/ShapeShapeprepare_tensors/Reshape*
out_type0*
_output_shapes
:*
T0
Ь
Qtrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputCtrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/Shapeconv_1/weights/W_conv_1/readStrain/gradients/conv_1/convolution_and_non_linear/add_grad/tuple/control_dependency*
use_cudnn_on_gpu(*
T0*
paddingSAME*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
data_formatNHWC*
strides


Etrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/Shape_1Const*%
valueB"         
   *
dtype0*
_output_shapes
:
Ї
Rtrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterprepare_tensors/ReshapeEtrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/Shape_1Strain/gradients/conv_1/convolution_and_non_linear/add_grad/tuple/control_dependency*
paddingSAME*
T0*
data_formatNHWC*
strides
*&
_output_shapes
:
*
use_cudnn_on_gpu(
џ
Ntrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/tuple/group_depsNoOpR^train/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInputS^train/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilter
Ј
Vtrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/tuple/control_dependencyIdentityQtrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInputO^train/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/tuple/group_deps*
T0*d
_classZ
XVloc:@train/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropInput*1
_output_shapes
:џџџџџџџџџ
Ё
Xtrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/tuple/control_dependency_1IdentityRtrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilterO^train/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/tuple/group_deps*e
_class[
YWloc:@train/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
:
*
T0

train/beta1_power/initial_valueConst*
valueB
 *fff?**
_class 
loc:@conv_1/weights/W_conv_1*
dtype0*
_output_shapes
: 
Ё
train/beta1_power
VariableV2*
	container *
dtype0**
_class 
loc:@conv_1/weights/W_conv_1*
_output_shapes
: *
shape: *
shared_name 
Ь
train/beta1_power/AssignAssigntrain/beta1_powertrain/beta1_power/initial_value*
use_locking(*
T0**
_class 
loc:@conv_1/weights/W_conv_1*
validate_shape(*
_output_shapes
: 

train/beta1_power/readIdentitytrain/beta1_power**
_class 
loc:@conv_1/weights/W_conv_1*
_output_shapes
: *
T0

train/beta2_power/initial_valueConst*
valueB
 *wО?**
_class 
loc:@conv_1/weights/W_conv_1*
dtype0*
_output_shapes
: 
Ё
train/beta2_power
VariableV2*
shared_name **
_class 
loc:@conv_1/weights/W_conv_1*
	container *
shape: *
dtype0*
_output_shapes
: 
Ь
train/beta2_power/AssignAssigntrain/beta2_powertrain/beta2_power/initial_value**
_class 
loc:@conv_1/weights/W_conv_1*
_output_shapes
: *
T0*
validate_shape(*
use_locking(

train/beta2_power/readIdentitytrain/beta2_power*
T0**
_class 
loc:@conv_1/weights/W_conv_1*
_output_shapes
: 
p
train/zerosConst*%
valueB
*    *
dtype0*&
_output_shapes
:

Ь
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

о
#conv_1/weights/W_conv_1/Adam/AssignAssignconv_1/weights/W_conv_1/Adamtrain/zeros*
use_locking(*
T0**
_class 
loc:@conv_1/weights/W_conv_1*
validate_shape(*&
_output_shapes
:

Ј
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
Ю
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
ф
%conv_1/weights/W_conv_1/Adam_1/AssignAssignconv_1/weights/W_conv_1/Adam_1train/zeros_1*
use_locking(*
T0**
_class 
loc:@conv_1/weights/W_conv_1*
validate_shape(*&
_output_shapes
:

Ќ
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
В
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
б
"conv_1/biases/b_conv_1/Adam/AssignAssignconv_1/biases/b_conv_1/Adamtrain/zeros_2*)
_class
loc:@conv_1/biases/b_conv_1*
_output_shapes
:
*
T0*
validate_shape(*
use_locking(

 conv_1/biases/b_conv_1/Adam/readIdentityconv_1/biases/b_conv_1/Adam*)
_class
loc:@conv_1/biases/b_conv_1*
_output_shapes
:
*
T0
Z
train/zeros_3Const*
valueB
*    *
_output_shapes
:
*
dtype0
Д
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
е
$conv_1/biases/b_conv_1/Adam_1/AssignAssignconv_1/biases/b_conv_1/Adam_1train/zeros_3*
use_locking(*
T0*)
_class
loc:@conv_1/biases/b_conv_1*
validate_shape(*
_output_shapes
:


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
Ь
conv_2/weights/W_conv_2/Adam
VariableV2*
shape:
*&
_output_shapes
:
*
shared_name **
_class 
loc:@conv_2/weights/W_conv_2*
dtype0*
	container 
р
#conv_2/weights/W_conv_2/Adam/AssignAssignconv_2/weights/W_conv_2/Adamtrain/zeros_4**
_class 
loc:@conv_2/weights/W_conv_2*&
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
Ј
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
Ю
conv_2/weights/W_conv_2/Adam_1
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
ф
%conv_2/weights/W_conv_2/Adam_1/AssignAssignconv_2/weights/W_conv_2/Adam_1train/zeros_5**
_class 
loc:@conv_2/weights/W_conv_2*&
_output_shapes
:
*
T0*
validate_shape(*
use_locking(
Ќ
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
В
conv_2/biases/b_conv_2/Adam
VariableV2*
	container *
dtype0*)
_class
loc:@conv_2/biases/b_conv_2*
_output_shapes
:*
shape:*
shared_name 
б
"conv_2/biases/b_conv_2/Adam/AssignAssignconv_2/biases/b_conv_2/Adamtrain/zeros_6*)
_class
loc:@conv_2/biases/b_conv_2*
_output_shapes
:*
T0*
validate_shape(*
use_locking(

 conv_2/biases/b_conv_2/Adam/readIdentityconv_2/biases/b_conv_2/Adam*)
_class
loc:@conv_2/biases/b_conv_2*
_output_shapes
:*
T0
Z
train/zeros_7Const*
valueB*    *
_output_shapes
:*
dtype0
Д
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
е
$conv_2/biases/b_conv_2/Adam_1/AssignAssignconv_2/biases/b_conv_2/Adam_1train/zeros_7*
use_locking(*
T0*)
_class
loc:@conv_2/biases/b_conv_2*
validate_shape(*
_output_shapes
:

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
Ь
conv_3/weights/W_conv_3/Adam
VariableV2*
shared_name **
_class 
loc:@conv_3/weights/W_conv_3*
	container *
shape:(*
dtype0*&
_output_shapes
:(
р
#conv_3/weights/W_conv_3/Adam/AssignAssignconv_3/weights/W_conv_3/Adamtrain/zeros_8*
use_locking(*
T0**
_class 
loc:@conv_3/weights/W_conv_3*
validate_shape(*&
_output_shapes
:(
Ј
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
Ю
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
ф
%conv_3/weights/W_conv_3/Adam_1/AssignAssignconv_3/weights/W_conv_3/Adam_1train/zeros_9**
_class 
loc:@conv_3/weights/W_conv_3*&
_output_shapes
:(*
T0*
validate_shape(*
use_locking(
Ќ
#conv_3/weights/W_conv_3/Adam_1/readIdentityconv_3/weights/W_conv_3/Adam_1**
_class 
loc:@conv_3/weights/W_conv_3*&
_output_shapes
:(*
T0
[
train/zeros_10Const*
valueB(*    *
dtype0*
_output_shapes
:(
В
conv_3/biases/b_conv_3/Adam
VariableV2*)
_class
loc:@conv_3/biases/b_conv_3*
_output_shapes
:(*
shape:(*
dtype0*
shared_name *
	container 
в
"conv_3/biases/b_conv_3/Adam/AssignAssignconv_3/biases/b_conv_3/Adamtrain/zeros_10*)
_class
loc:@conv_3/biases/b_conv_3*
_output_shapes
:(*
T0*
validate_shape(*
use_locking(

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
Д
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
ж
$conv_3/biases/b_conv_3/Adam_1/AssignAssignconv_3/biases/b_conv_3/Adam_1train/zeros_11*
use_locking(*
T0*)
_class
loc:@conv_3/biases/b_conv_3*
validate_shape(*
_output_shapes
:(

"conv_3/biases/b_conv_3/Adam_1/readIdentityconv_3/biases/b_conv_3/Adam_1*
T0*)
_class
loc:@conv_3/biases/b_conv_3*
_output_shapes
:(
e
train/zeros_12Const*
valueB	d*    *
dtype0*
_output_shapes
:	d
Д
fc_1/weights/W_fc1/Adam
VariableV2*
shape:	d*
_output_shapes
:	d*
shared_name *%
_class
loc:@fc_1/weights/W_fc1*
dtype0*
	container 
Ы
fc_1/weights/W_fc1/Adam/AssignAssignfc_1/weights/W_fc1/Adamtrain/zeros_12*%
_class
loc:@fc_1/weights/W_fc1*
_output_shapes
:	d*
T0*
validate_shape(*
use_locking(

fc_1/weights/W_fc1/Adam/readIdentityfc_1/weights/W_fc1/Adam*%
_class
loc:@fc_1/weights/W_fc1*
_output_shapes
:	d*
T0
e
train/zeros_13Const*
valueB	d*    *
_output_shapes
:	d*
dtype0
Ж
fc_1/weights/W_fc1/Adam_1
VariableV2*
shape:	d*
_output_shapes
:	d*
shared_name *%
_class
loc:@fc_1/weights/W_fc1*
dtype0*
	container 
Я
 fc_1/weights/W_fc1/Adam_1/AssignAssignfc_1/weights/W_fc1/Adam_1train/zeros_13*
use_locking(*
T0*%
_class
loc:@fc_1/weights/W_fc1*
validate_shape(*
_output_shapes
:	d

fc_1/weights/W_fc1/Adam_1/readIdentityfc_1/weights/W_fc1/Adam_1*%
_class
loc:@fc_1/weights/W_fc1*
_output_shapes
:	d*
T0
[
train/zeros_14Const*
valueBd*    *
dtype0*
_output_shapes
:d
Ј
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
У
fc_1/biases/b_fc1/Adam/AssignAssignfc_1/biases/b_fc1/Adamtrain/zeros_14*
use_locking(*
T0*$
_class
loc:@fc_1/biases/b_fc1*
validate_shape(*
_output_shapes
:d

fc_1/biases/b_fc1/Adam/readIdentityfc_1/biases/b_fc1/Adam*$
_class
loc:@fc_1/biases/b_fc1*
_output_shapes
:d*
T0
[
train/zeros_15Const*
valueBd*    *
dtype0*
_output_shapes
:d
Њ
fc_1/biases/b_fc1/Adam_1
VariableV2*
shared_name *$
_class
loc:@fc_1/biases/b_fc1*
	container *
shape:d*
dtype0*
_output_shapes
:d
Ч
fc_1/biases/b_fc1/Adam_1/AssignAssignfc_1/biases/b_fc1/Adam_1train/zeros_15*$
_class
loc:@fc_1/biases/b_fc1*
_output_shapes
:d*
T0*
validate_shape(*
use_locking(

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
В
fc_2/weights/W_fc2/Adam
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
Ъ
fc_2/weights/W_fc2/Adam/AssignAssignfc_2/weights/W_fc2/Adamtrain/zeros_16*
use_locking(*
T0*%
_class
loc:@fc_2/weights/W_fc2*
validate_shape(*
_output_shapes

:d

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
Д
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
Ю
 fc_2/weights/W_fc2/Adam_1/AssignAssignfc_2/weights/W_fc2/Adam_1train/zeros_17*%
_class
loc:@fc_2/weights/W_fc2*
_output_shapes

:d*
T0*
validate_shape(*
use_locking(

fc_2/weights/W_fc2/Adam_1/readIdentityfc_2/weights/W_fc2/Adam_1*%
_class
loc:@fc_2/weights/W_fc2*
_output_shapes

:d*
T0
[
train/zeros_18Const*
valueB*    *
_output_shapes
:*
dtype0
Њ
fc_2/biases/b_fc_2/Adam
VariableV2*
shared_name *%
_class
loc:@fc_2/biases/b_fc_2*
	container *
shape:*
dtype0*
_output_shapes
:
Ц
fc_2/biases/b_fc_2/Adam/AssignAssignfc_2/biases/b_fc_2/Adamtrain/zeros_18*
use_locking(*
T0*%
_class
loc:@fc_2/biases/b_fc_2*
validate_shape(*
_output_shapes
:

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
Ќ
fc_2/biases/b_fc_2/Adam_1
VariableV2*%
_class
loc:@fc_2/biases/b_fc_2*
_output_shapes
:*
shape:*
dtype0*
shared_name *
	container 
Ъ
 fc_2/biases/b_fc_2/Adam_1/AssignAssignfc_2/biases/b_fc_2/Adam_1train/zeros_19*
use_locking(*
T0*%
_class
loc:@fc_2/biases/b_fc_2*
validate_shape(*
_output_shapes
:

fc_2/biases/b_fc_2/Adam_1/readIdentityfc_2/biases/b_fc_2/Adam_1*
T0*%
_class
loc:@fc_2/biases/b_fc_2*
_output_shapes
:
]
train/Adam/learning_rateConst*
valueB
 *Зб8*
dtype0*
_output_shapes
: 
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
 *wО?*
dtype0*
_output_shapes
: 
W
train/Adam/epsilonConst*
valueB
 *wЬ+2*
_output_shapes
: *
dtype0
у
3train/Adam/update_conv_1/weights/W_conv_1/ApplyAdam	ApplyAdamconv_1/weights/W_conv_1conv_1/weights/W_conv_1/Adamconv_1/weights/W_conv_1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonXtrain/gradients/conv_1/convolution_and_non_linear/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0**
_class 
loc:@conv_1/weights/W_conv_1*&
_output_shapes
:

Я
2train/Adam/update_conv_1/biases/b_conv_1/ApplyAdam	ApplyAdamconv_1/biases/b_conv_1conv_1/biases/b_conv_1/Adamconv_1/biases/b_conv_1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonUtrain/gradients/conv_1/convolution_and_non_linear/add_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@conv_1/biases/b_conv_1*
_output_shapes
:

у
3train/Adam/update_conv_2/weights/W_conv_2/ApplyAdam	ApplyAdamconv_2/weights/W_conv_2conv_2/weights/W_conv_2/Adamconv_2/weights/W_conv_2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonXtrain/gradients/conv_2/convolution_and_non_linear/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0**
_class 
loc:@conv_2/weights/W_conv_2*&
_output_shapes
:

Я
2train/Adam/update_conv_2/biases/b_conv_2/ApplyAdam	ApplyAdamconv_2/biases/b_conv_2conv_2/biases/b_conv_2/Adamconv_2/biases/b_conv_2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonUtrain/gradients/conv_2/convolution_and_non_linear/add_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@conv_2/biases/b_conv_2*
_output_shapes
:
у
3train/Adam/update_conv_3/weights/W_conv_3/ApplyAdam	ApplyAdamconv_3/weights/W_conv_3conv_3/weights/W_conv_3/Adamconv_3/weights/W_conv_3/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonXtrain/gradients/conv_3/convolution_and_non_linear/Conv2D_grad/tuple/control_dependency_1**
_class 
loc:@conv_3/weights/W_conv_3*&
_output_shapes
:(*
T0*
use_locking( 
Я
2train/Adam/update_conv_3/biases/b_conv_3/ApplyAdam	ApplyAdamconv_3/biases/b_conv_3conv_3/biases/b_conv_3/Adamconv_3/biases/b_conv_3/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonUtrain/gradients/conv_3/convolution_and_non_linear/add_grad/tuple/control_dependency_1*)
_class
loc:@conv_3/biases/b_conv_3*
_output_shapes
:(*
T0*
use_locking( 
М
.train/Adam/update_fc_1/weights/W_fc1/ApplyAdam	ApplyAdamfc_1/weights/W_fc1fc_1/weights/W_fc1/Adamfc_1/weights/W_fc1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonQtrain/gradients/fc_1/matmul_and_non_linear/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*%
_class
loc:@fc_1/weights/W_fc1*
_output_shapes
:	d
Џ
-train/Adam/update_fc_1/biases/b_fc1/ApplyAdam	ApplyAdamfc_1/biases/b_fc1fc_1/biases/b_fc1/Adamfc_1/biases/b_fc1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilonNtrain/gradients/fc_1/matmul_and_non_linear/add_grad/tuple/control_dependency_1*$
_class
loc:@fc_1/biases/b_fc1*
_output_shapes
:d*
T0*
use_locking( 
Ј
.train/Adam/update_fc_2/weights/W_fc2/ApplyAdam	ApplyAdamfc_2/weights/W_fc2fc_2/weights/W_fc2/Adamfc_2/weights/W_fc2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/outputs/MatMul_grad/tuple/control_dependency_1*%
_class
loc:@fc_2/weights/W_fc2*
_output_shapes

:d*
T0*
use_locking( 
Ё
.train/Adam/update_fc_2/biases/b_fc_2/ApplyAdam	ApplyAdamfc_2/biases/b_fc_2fc_2/biases/b_fc_2/Adamfc_2/biases/b_fc_2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/outputs/add_grad/tuple/control_dependency_1*%
_class
loc:@fc_2/biases/b_fc_2*
_output_shapes
:*
T0*
use_locking( 

train/Adam/mulMultrain/beta1_power/readtrain/Adam/beta14^train/Adam/update_conv_1/weights/W_conv_1/ApplyAdam3^train/Adam/update_conv_1/biases/b_conv_1/ApplyAdam4^train/Adam/update_conv_2/weights/W_conv_2/ApplyAdam3^train/Adam/update_conv_2/biases/b_conv_2/ApplyAdam4^train/Adam/update_conv_3/weights/W_conv_3/ApplyAdam3^train/Adam/update_conv_3/biases/b_conv_3/ApplyAdam/^train/Adam/update_fc_1/weights/W_fc1/ApplyAdam.^train/Adam/update_fc_1/biases/b_fc1/ApplyAdam/^train/Adam/update_fc_2/weights/W_fc2/ApplyAdam/^train/Adam/update_fc_2/biases/b_fc_2/ApplyAdam*
T0**
_class 
loc:@conv_1/weights/W_conv_1*
_output_shapes
: 
Д
train/Adam/AssignAssigntrain/beta1_powertrain/Adam/mul*
use_locking( *
T0**
_class 
loc:@conv_1/weights/W_conv_1*
validate_shape(*
_output_shapes
: 

train/Adam/mul_1Multrain/beta2_power/readtrain/Adam/beta24^train/Adam/update_conv_1/weights/W_conv_1/ApplyAdam3^train/Adam/update_conv_1/biases/b_conv_1/ApplyAdam4^train/Adam/update_conv_2/weights/W_conv_2/ApplyAdam3^train/Adam/update_conv_2/biases/b_conv_2/ApplyAdam4^train/Adam/update_conv_3/weights/W_conv_3/ApplyAdam3^train/Adam/update_conv_3/biases/b_conv_3/ApplyAdam/^train/Adam/update_fc_1/weights/W_fc1/ApplyAdam.^train/Adam/update_fc_1/biases/b_fc1/ApplyAdam/^train/Adam/update_fc_2/weights/W_fc2/ApplyAdam/^train/Adam/update_fc_2/biases/b_fc_2/ApplyAdam**
_class 
loc:@conv_1/weights/W_conv_1*
_output_shapes
: *
T0
И
train/Adam/Assign_1Assigntrain/beta2_powertrain/Adam/mul_1*
use_locking( *
T0**
_class 
loc:@conv_1/weights/W_conv_1*
validate_shape(*
_output_shapes
: 
Р

train/AdamNoOp4^train/Adam/update_conv_1/weights/W_conv_1/ApplyAdam3^train/Adam/update_conv_1/biases/b_conv_1/ApplyAdam4^train/Adam/update_conv_2/weights/W_conv_2/ApplyAdam3^train/Adam/update_conv_2/biases/b_conv_2/ApplyAdam4^train/Adam/update_conv_3/weights/W_conv_3/ApplyAdam3^train/Adam/update_conv_3/biases/b_conv_3/ApplyAdam/^train/Adam/update_fc_1/weights/W_fc1/ApplyAdam.^train/Adam/update_fc_1/biases/b_fc1/ApplyAdam/^train/Adam/update_fc_2/weights/W_fc2/ApplyAdam/^train/Adam/update_fc_2/biases/b_fc_2/ApplyAdam^train/Adam/Assign^train/Adam/Assign_1
[
evaluate/ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 

evaluate/ArgMaxArgMaxoutputs/Softmaxevaluate/ArgMax/dimension*

Tidx0*
T0*#
_output_shapes
:џџџџџџџџџ
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
:џџџџџџџџџ
i
evaluate/EqualEqualevaluate/ArgMaxevaluate/ArgMax_1*
T0	*#
_output_shapes
:џџџџџџџџџ
b
evaluate/CastCastevaluate/Equal*#
_output_shapes
:џџџџџџџџџ*

DstT0*

SrcT0

X
evaluate/ConstConst*
valueB: *
_output_shapes
:*
dtype0
r
evaluate/MeanMeanevaluate/Castevaluate/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
`
Merge/MergeSummaryMergeSummarycross_entorpy/cross_entorpy*
_output_shapes
: *
N"".
	summaries!

cross_entorpy/cross_entorpy:0"ѕ
trainable_variablesнк
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


train/Adam"
	variables
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
fc_2/biases/b_fc_2/Adam_1:0 fc_2/biases/b_fc_2/Adam_1/Assign fc_2/biases/b_fc_2/Adam_1/read:0LЬџ"/       m]P	ж9чБBжA*$
"
cross_entorpy/cross_entorpy,ћI?.эЃ1       щ	MРЕВBжA*$
"
cross_entorpy/cross_entorpy?bрЉN1       щ	ѓU0(ВBжA(*$
"
cross_entorpy/cross_entorpy/э№>Q21       щ	'КHВBжA<*$
"
cross_entorpy/cross_entorpyЅXу>TЬPЫ1       щ	Уђ:iВBжAP*$
"
cross_entorpy/cross_entorpy й>|Ї\1       щ	%zЏВBжAd*$
"
cross_entorpy/cross_entorpyд3Ш>ТR<1       щ	~ЊВBжAx*$
"
cross_entorpy/cross_entorpy}*Д>ы^p2       $Vь	ШЁЪВBжA*$
"
cross_entorpy/cross_entorpy;Х>­Ўюд2       $Vь	@ыВBжA *$
"
cross_entorpy/cross_entorpyQЊ>гУ ­2       $Vь	hГBжAД*$
"
cross_entorpy/cross_entorpyaЌ>>x2       $Vь	Еѓ+ГBжAШ*$
"
cross_entorpy/cross_entorpy0>Њ2       $Vь	`LГBжAм*$
"
cross_entorpy/cross_entorpyu>ѓ\2       $Vь	x{лlГBжA№*$
"
cross_entorpy/cross_entorpyІ{>$нЋл2       $Vь	хUГBжA*$
"
cross_entorpy/cross_entorpyпЊk>Я2       $Vь	xAЮ­ГBжA*$
"
cross_entorpy/cross_entorpyXQX>ТDKЛ2       $Vь	NdSЮГBжAЌ*$
"
cross_entorpy/cross_entorpylH>кrђЖ2       $Vь	
ѓЯюГBжAР*$
"
cross_entorpy/cross_entorpyZ)>|9Iю2       $Vь	цZ?ДBжAд*$
"
cross_entorpy/cross_entorpyм(>Xi+2       $Vь	X8Е/ДBжAш*$
"
cross_entorpy/cross_entorpyБ>/В~2       $Vь	Ј0PДBжAќ*$
"
cross_entorpy/cross_entorpy,тъ=Џ2       $Vь	'нЛpДBжA*$
"
cross_entorpy/cross_entorpyъі=№МЈ2       $Vь	Р$ДBжAЄ*$
"
cross_entorpy/cross_entorpyDи=
c2       $Vь	uЫБДBжAИ*$
"
cross_entorpy/cross_entorpy9qЃ=ќРc
2       $Vь	,вДBжAЬ*$
"
cross_entorpy/cross_entorpysyЄ=іEЯ"2       $Vь	
іђДBжAр*$
"
cross_entorpy/cross_entorpysВ=(о2       $Vь	ђЕBжAє*$
"
cross_entorpy/cross_entorpyч=е&У2       $Vь	'кx3ЕBжA*$
"
cross_entorpy/cross_entorpyn`=:zЧ2       $Vь	ЌўцSЕBжA*$
"
cross_entorpy/cross_entorpyЄO=B	d2       $Vь	mQtЕBжAА*$
"
cross_entorpy/cross_entorpyГё=Ј/V2       $Vь	2ЬЪЕBжAФ*$
"
cross_entorpy/cross_entorpyM<=аp2       $Vь	іQЕЕBжAи*$
"
cross_entorpy/cross_entorpyf}
=0Ц2       $Vь	ЕеЕBжAь*$
"
cross_entorpy/cross_entorpyM9ј<<Ъw'2       $Vь	3міЕBжA*$
"
cross_entorpy/cross_entorpyЯGф<зЃВ{2       $Vь	-ЭЖBжA*$
"
cross_entorpy/cross_entorpyAх<eq92       $Vь	­7ЖBжAЈ*$
"
cross_entorpy/cross_entorpyџ(Ћ<р;rж2       $Vь	РьxWЖBжAМ*$
"
cross_entorpy/cross_entorpyЁГЋ<иѓP2       $Vь	ч4ъwЖBжAа*$
"
cross_entorpy/cross_entorpyэ<МЃЖ2       $Vь	_mЖBжAф*$
"
cross_entorpy/cross_entorpyb<cБX2       $Vь	eћлИЖBжAј*$
"
cross_entorpy/cross_entorpy=<М­2       $Vь	ОMSйЖBжA*$
"
cross_entorpy/cross_entorpyуU<Ц&2       $Vь	ЯљЖBжA *$
"
cross_entorpy/cross_entorpyмI<u	bA2       $Vь	Пe6ЗBжAД*$
"
cross_entorpy/cross_entorpyУ G<mK|2       $Vь	 'І:ЗBжAШ*$
"
cross_entorpy/cross_entorpyaш<Џо2       $Vь	u [ЗBжAм*$
"
cross_entorpy/cross_entorpyЃ<Упa2       $Vь	Eѕ{ЗBжA№*$
"
cross_entorpy/cross_entorpyп<Q 2       $Vь	ъВЗBжA*$
"
cross_entorpy/cross_entorpyiЭ<ш$Ъ2       $Vь	vУtМЗBжA*$
"
cross_entorpy/cross_entorpyHЎї;;џ2       $Vь	xЕAнЗBжAЌ*$
"
cross_entorpy/cross_entorpyзУ;0<	П2       $Vь	 ўЗBжAР*$
"
cross_entorpy/cross_entorpyЬ5и;ўА2       $Vь	4ИBжAд*$
"
cross_entorpy/cross_entorpyФБН;  ОZ2       $Vь	
ј>ИBжAш*$
"
cross_entorpy/cross_entorpyюЇ;?ФўО