╗▀
М'▐&
:
Add
x"T
y"T
z"T"
Ttype:
2	
A
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
Ы
ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
E
AssignAddVariableOp
resource
value"dtype"
dtypetypeИ
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
Ъ
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

8
DivNoNan
x"T
y"T
z"T"
Ttype:	
2
h
Equal
x"T
y"T
z
"
Ttype:
2	
"$
incompatible_shape_errorbool(Р
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
?

LogSoftmax
logits"T

logsoftmax"T"
Ttype:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
╘
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
6
Pow
x"T
y"T
z"T"
Ttype:

2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
р
ResourceApplyAdam
var
m
v
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( И
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
j
SoftmaxCrossEntropyWithLogits
features"T
labels"T	
loss"T
backprop"T"
Ttype:
2
-
Sqrt
x"T
y"T"
Ttype:

2
Ў
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
М
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
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
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ
9
VarIsInitializedOp
resource
is_initialized
И
&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.15.02v1.15.0-rc3-22-g590d6eef7e╣┬
~
input_1Placeholder*&
shape:         рр*
dtype0*1
_output_shapes
:         рр
╡
4block1_conv1/kernel/Initializer/random_uniform/shapeConst*%
valueB"         @   *&
_class
loc:@block1_conv1/kernel*
dtype0*
_output_shapes
:
Я
2block1_conv1/kernel/Initializer/random_uniform/minConst*
valueB
 *8J╠╜*&
_class
loc:@block1_conv1/kernel*
dtype0*
_output_shapes
: 
Я
2block1_conv1/kernel/Initializer/random_uniform/maxConst*
valueB
 *8J╠=*&
_class
loc:@block1_conv1/kernel*
dtype0*
_output_shapes
: 
В
<block1_conv1/kernel/Initializer/random_uniform/RandomUniformRandomUniform4block1_conv1/kernel/Initializer/random_uniform/shape*

seed *
T0*&
_class
loc:@block1_conv1/kernel*
seed2 *
dtype0*&
_output_shapes
:@
ъ
2block1_conv1/kernel/Initializer/random_uniform/subSub2block1_conv1/kernel/Initializer/random_uniform/max2block1_conv1/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@block1_conv1/kernel*
_output_shapes
: 
Д
2block1_conv1/kernel/Initializer/random_uniform/mulMul<block1_conv1/kernel/Initializer/random_uniform/RandomUniform2block1_conv1/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@block1_conv1/kernel*&
_output_shapes
:@
Ў
.block1_conv1/kernel/Initializer/random_uniformAdd2block1_conv1/kernel/Initializer/random_uniform/mul2block1_conv1/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@block1_conv1/kernel*&
_output_shapes
:@
├
block1_conv1/kernelVarHandleOp*$
shared_nameblock1_conv1/kernel*&
_class
loc:@block1_conv1/kernel*
	container *
shape:@*
dtype0*
_output_shapes
: 
w
4block1_conv1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock1_conv1/kernel*
_output_shapes
: 
А
block1_conv1/kernel/AssignAssignVariableOpblock1_conv1/kernel.block1_conv1/kernel/Initializer/random_uniform*
dtype0
Г
'block1_conv1/kernel/Read/ReadVariableOpReadVariableOpblock1_conv1/kernel*
dtype0*&
_output_shapes
:@
Ц
#block1_conv1/bias/Initializer/zerosConst*
valueB@*    *$
_class
loc:@block1_conv1/bias*
dtype0*
_output_shapes
:@
▒
block1_conv1/biasVarHandleOp*"
shared_nameblock1_conv1/bias*$
_class
loc:@block1_conv1/bias*
	container *
shape:@*
dtype0*
_output_shapes
: 
s
2block1_conv1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock1_conv1/bias*
_output_shapes
: 
q
block1_conv1/bias/AssignAssignVariableOpblock1_conv1/bias#block1_conv1/bias/Initializer/zeros*
dtype0
s
%block1_conv1/bias/Read/ReadVariableOpReadVariableOpblock1_conv1/bias*
dtype0*
_output_shapes
:@
k
block1_conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
~
"block1_conv1/Conv2D/ReadVariableOpReadVariableOpblock1_conv1/kernel*
dtype0*&
_output_shapes
:@
Н
block1_conv1/Conv2DConv2Dinput_1"block1_conv1/Conv2D/ReadVariableOp*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:         рр@
q
#block1_conv1/BiasAdd/ReadVariableOpReadVariableOpblock1_conv1/bias*
dtype0*
_output_shapes
:@
м
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D#block1_conv1/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*1
_output_shapes
:         рр@
k
block1_conv1/ReluRelublock1_conv1/BiasAdd*
T0*1
_output_shapes
:         рр@
╡
4block1_conv2/kernel/Initializer/random_uniform/shapeConst*%
valueB"      @   @   *&
_class
loc:@block1_conv2/kernel*
dtype0*
_output_shapes
:
Я
2block1_conv2/kernel/Initializer/random_uniform/minConst*
valueB
 *:═У╜*&
_class
loc:@block1_conv2/kernel*
dtype0*
_output_shapes
: 
Я
2block1_conv2/kernel/Initializer/random_uniform/maxConst*
valueB
 *:═У=*&
_class
loc:@block1_conv2/kernel*
dtype0*
_output_shapes
: 
В
<block1_conv2/kernel/Initializer/random_uniform/RandomUniformRandomUniform4block1_conv2/kernel/Initializer/random_uniform/shape*

seed *
T0*&
_class
loc:@block1_conv2/kernel*
seed2 *
dtype0*&
_output_shapes
:@@
ъ
2block1_conv2/kernel/Initializer/random_uniform/subSub2block1_conv2/kernel/Initializer/random_uniform/max2block1_conv2/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@block1_conv2/kernel*
_output_shapes
: 
Д
2block1_conv2/kernel/Initializer/random_uniform/mulMul<block1_conv2/kernel/Initializer/random_uniform/RandomUniform2block1_conv2/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@block1_conv2/kernel*&
_output_shapes
:@@
Ў
.block1_conv2/kernel/Initializer/random_uniformAdd2block1_conv2/kernel/Initializer/random_uniform/mul2block1_conv2/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@block1_conv2/kernel*&
_output_shapes
:@@
├
block1_conv2/kernelVarHandleOp*$
shared_nameblock1_conv2/kernel*&
_class
loc:@block1_conv2/kernel*
	container *
shape:@@*
dtype0*
_output_shapes
: 
w
4block1_conv2/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock1_conv2/kernel*
_output_shapes
: 
А
block1_conv2/kernel/AssignAssignVariableOpblock1_conv2/kernel.block1_conv2/kernel/Initializer/random_uniform*
dtype0
Г
'block1_conv2/kernel/Read/ReadVariableOpReadVariableOpblock1_conv2/kernel*
dtype0*&
_output_shapes
:@@
Ц
#block1_conv2/bias/Initializer/zerosConst*
valueB@*    *$
_class
loc:@block1_conv2/bias*
dtype0*
_output_shapes
:@
▒
block1_conv2/biasVarHandleOp*"
shared_nameblock1_conv2/bias*$
_class
loc:@block1_conv2/bias*
	container *
shape:@*
dtype0*
_output_shapes
: 
s
2block1_conv2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock1_conv2/bias*
_output_shapes
: 
q
block1_conv2/bias/AssignAssignVariableOpblock1_conv2/bias#block1_conv2/bias/Initializer/zeros*
dtype0
s
%block1_conv2/bias/Read/ReadVariableOpReadVariableOpblock1_conv2/bias*
dtype0*
_output_shapes
:@
k
block1_conv2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
~
"block1_conv2/Conv2D/ReadVariableOpReadVariableOpblock1_conv2/kernel*
dtype0*&
_output_shapes
:@@
Ч
block1_conv2/Conv2DConv2Dblock1_conv1/Relu"block1_conv2/Conv2D/ReadVariableOp*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*1
_output_shapes
:         рр@
q
#block1_conv2/BiasAdd/ReadVariableOpReadVariableOpblock1_conv2/bias*
dtype0*
_output_shapes
:@
м
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D#block1_conv2/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*1
_output_shapes
:         рр@
k
block1_conv2/ReluRelublock1_conv2/BiasAdd*
T0*1
_output_shapes
:         рр@
╛
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*/
_output_shapes
:         pp@
╡
4block2_conv1/kernel/Initializer/random_uniform/shapeConst*%
valueB"      @   А   *&
_class
loc:@block2_conv1/kernel*
dtype0*
_output_shapes
:
Я
2block2_conv1/kernel/Initializer/random_uniform/minConst*
valueB
 *я[q╜*&
_class
loc:@block2_conv1/kernel*
dtype0*
_output_shapes
: 
Я
2block2_conv1/kernel/Initializer/random_uniform/maxConst*
valueB
 *я[q=*&
_class
loc:@block2_conv1/kernel*
dtype0*
_output_shapes
: 
Г
<block2_conv1/kernel/Initializer/random_uniform/RandomUniformRandomUniform4block2_conv1/kernel/Initializer/random_uniform/shape*

seed *
T0*&
_class
loc:@block2_conv1/kernel*
seed2 *
dtype0*'
_output_shapes
:@А
ъ
2block2_conv1/kernel/Initializer/random_uniform/subSub2block2_conv1/kernel/Initializer/random_uniform/max2block2_conv1/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@block2_conv1/kernel*
_output_shapes
: 
Е
2block2_conv1/kernel/Initializer/random_uniform/mulMul<block2_conv1/kernel/Initializer/random_uniform/RandomUniform2block2_conv1/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@block2_conv1/kernel*'
_output_shapes
:@А
ў
.block2_conv1/kernel/Initializer/random_uniformAdd2block2_conv1/kernel/Initializer/random_uniform/mul2block2_conv1/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@block2_conv1/kernel*'
_output_shapes
:@А
─
block2_conv1/kernelVarHandleOp*$
shared_nameblock2_conv1/kernel*&
_class
loc:@block2_conv1/kernel*
	container *
shape:@А*
dtype0*
_output_shapes
: 
w
4block2_conv1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock2_conv1/kernel*
_output_shapes
: 
А
block2_conv1/kernel/AssignAssignVariableOpblock2_conv1/kernel.block2_conv1/kernel/Initializer/random_uniform*
dtype0
Д
'block2_conv1/kernel/Read/ReadVariableOpReadVariableOpblock2_conv1/kernel*
dtype0*'
_output_shapes
:@А
Ш
#block2_conv1/bias/Initializer/zerosConst*
valueBА*    *$
_class
loc:@block2_conv1/bias*
dtype0*
_output_shapes	
:А
▓
block2_conv1/biasVarHandleOp*"
shared_nameblock2_conv1/bias*$
_class
loc:@block2_conv1/bias*
	container *
shape:А*
dtype0*
_output_shapes
: 
s
2block2_conv1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock2_conv1/bias*
_output_shapes
: 
q
block2_conv1/bias/AssignAssignVariableOpblock2_conv1/bias#block2_conv1/bias/Initializer/zeros*
dtype0
t
%block2_conv1/bias/Read/ReadVariableOpReadVariableOpblock2_conv1/bias*
dtype0*
_output_shapes	
:А
k
block2_conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:

"block2_conv1/Conv2D/ReadVariableOpReadVariableOpblock2_conv1/kernel*
dtype0*'
_output_shapes
:@А
Ш
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool"block2_conv1/Conv2D/ReadVariableOp*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:         ppА
r
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOpblock2_conv1/bias*
dtype0*
_output_shapes	
:А
л
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D#block2_conv1/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*0
_output_shapes
:         ppА
j
block2_conv1/ReluRelublock2_conv1/BiasAdd*
T0*0
_output_shapes
:         ppА
╡
4block2_conv2/kernel/Initializer/random_uniform/shapeConst*%
valueB"      А   А   *&
_class
loc:@block2_conv2/kernel*
dtype0*
_output_shapes
:
Я
2block2_conv2/kernel/Initializer/random_uniform/minConst*
valueB
 *ьQ╜*&
_class
loc:@block2_conv2/kernel*
dtype0*
_output_shapes
: 
Я
2block2_conv2/kernel/Initializer/random_uniform/maxConst*
valueB
 *ьQ=*&
_class
loc:@block2_conv2/kernel*
dtype0*
_output_shapes
: 
Д
<block2_conv2/kernel/Initializer/random_uniform/RandomUniformRandomUniform4block2_conv2/kernel/Initializer/random_uniform/shape*

seed *
T0*&
_class
loc:@block2_conv2/kernel*
seed2 *
dtype0*(
_output_shapes
:АА
ъ
2block2_conv2/kernel/Initializer/random_uniform/subSub2block2_conv2/kernel/Initializer/random_uniform/max2block2_conv2/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@block2_conv2/kernel*
_output_shapes
: 
Ж
2block2_conv2/kernel/Initializer/random_uniform/mulMul<block2_conv2/kernel/Initializer/random_uniform/RandomUniform2block2_conv2/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@block2_conv2/kernel*(
_output_shapes
:АА
°
.block2_conv2/kernel/Initializer/random_uniformAdd2block2_conv2/kernel/Initializer/random_uniform/mul2block2_conv2/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@block2_conv2/kernel*(
_output_shapes
:АА
┼
block2_conv2/kernelVarHandleOp*$
shared_nameblock2_conv2/kernel*&
_class
loc:@block2_conv2/kernel*
	container *
shape:АА*
dtype0*
_output_shapes
: 
w
4block2_conv2/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock2_conv2/kernel*
_output_shapes
: 
А
block2_conv2/kernel/AssignAssignVariableOpblock2_conv2/kernel.block2_conv2/kernel/Initializer/random_uniform*
dtype0
Е
'block2_conv2/kernel/Read/ReadVariableOpReadVariableOpblock2_conv2/kernel*
dtype0*(
_output_shapes
:АА
Ш
#block2_conv2/bias/Initializer/zerosConst*
valueBА*    *$
_class
loc:@block2_conv2/bias*
dtype0*
_output_shapes	
:А
▓
block2_conv2/biasVarHandleOp*"
shared_nameblock2_conv2/bias*$
_class
loc:@block2_conv2/bias*
	container *
shape:А*
dtype0*
_output_shapes
: 
s
2block2_conv2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock2_conv2/bias*
_output_shapes
: 
q
block2_conv2/bias/AssignAssignVariableOpblock2_conv2/bias#block2_conv2/bias/Initializer/zeros*
dtype0
t
%block2_conv2/bias/Read/ReadVariableOpReadVariableOpblock2_conv2/bias*
dtype0*
_output_shapes	
:А
k
block2_conv2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
А
"block2_conv2/Conv2D/ReadVariableOpReadVariableOpblock2_conv2/kernel*
dtype0*(
_output_shapes
:АА
Ц
block2_conv2/Conv2DConv2Dblock2_conv1/Relu"block2_conv2/Conv2D/ReadVariableOp*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:         ppА
r
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOpblock2_conv2/bias*
dtype0*
_output_shapes	
:А
л
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D#block2_conv2/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*0
_output_shapes
:         ppА
j
block2_conv2/ReluRelublock2_conv2/BiasAdd*
T0*0
_output_shapes
:         ppА
┐
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*0
_output_shapes
:         88А
╡
4block3_conv1/kernel/Initializer/random_uniform/shapeConst*%
valueB"      А      *&
_class
loc:@block3_conv1/kernel*
dtype0*
_output_shapes
:
Я
2block3_conv1/kernel/Initializer/random_uniform/minConst*
valueB
 *лк*╜*&
_class
loc:@block3_conv1/kernel*
dtype0*
_output_shapes
: 
Я
2block3_conv1/kernel/Initializer/random_uniform/maxConst*
valueB
 *лк*=*&
_class
loc:@block3_conv1/kernel*
dtype0*
_output_shapes
: 
Д
<block3_conv1/kernel/Initializer/random_uniform/RandomUniformRandomUniform4block3_conv1/kernel/Initializer/random_uniform/shape*

seed *
T0*&
_class
loc:@block3_conv1/kernel*
seed2 *
dtype0*(
_output_shapes
:АА
ъ
2block3_conv1/kernel/Initializer/random_uniform/subSub2block3_conv1/kernel/Initializer/random_uniform/max2block3_conv1/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@block3_conv1/kernel*
_output_shapes
: 
Ж
2block3_conv1/kernel/Initializer/random_uniform/mulMul<block3_conv1/kernel/Initializer/random_uniform/RandomUniform2block3_conv1/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@block3_conv1/kernel*(
_output_shapes
:АА
°
.block3_conv1/kernel/Initializer/random_uniformAdd2block3_conv1/kernel/Initializer/random_uniform/mul2block3_conv1/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@block3_conv1/kernel*(
_output_shapes
:АА
┼
block3_conv1/kernelVarHandleOp*$
shared_nameblock3_conv1/kernel*&
_class
loc:@block3_conv1/kernel*
	container *
shape:АА*
dtype0*
_output_shapes
: 
w
4block3_conv1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock3_conv1/kernel*
_output_shapes
: 
А
block3_conv1/kernel/AssignAssignVariableOpblock3_conv1/kernel.block3_conv1/kernel/Initializer/random_uniform*
dtype0
Е
'block3_conv1/kernel/Read/ReadVariableOpReadVariableOpblock3_conv1/kernel*
dtype0*(
_output_shapes
:АА
Ш
#block3_conv1/bias/Initializer/zerosConst*
valueBА*    *$
_class
loc:@block3_conv1/bias*
dtype0*
_output_shapes	
:А
▓
block3_conv1/biasVarHandleOp*"
shared_nameblock3_conv1/bias*$
_class
loc:@block3_conv1/bias*
	container *
shape:А*
dtype0*
_output_shapes
: 
s
2block3_conv1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock3_conv1/bias*
_output_shapes
: 
q
block3_conv1/bias/AssignAssignVariableOpblock3_conv1/bias#block3_conv1/bias/Initializer/zeros*
dtype0
t
%block3_conv1/bias/Read/ReadVariableOpReadVariableOpblock3_conv1/bias*
dtype0*
_output_shapes	
:А
k
block3_conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
А
"block3_conv1/Conv2D/ReadVariableOpReadVariableOpblock3_conv1/kernel*
dtype0*(
_output_shapes
:АА
Ш
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool"block3_conv1/Conv2D/ReadVariableOp*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:         88А
r
#block3_conv1/BiasAdd/ReadVariableOpReadVariableOpblock3_conv1/bias*
dtype0*
_output_shapes	
:А
л
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D#block3_conv1/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*0
_output_shapes
:         88А
j
block3_conv1/ReluRelublock3_conv1/BiasAdd*
T0*0
_output_shapes
:         88А
╡
4block3_conv2/kernel/Initializer/random_uniform/shapeConst*%
valueB"            *&
_class
loc:@block3_conv2/kernel*
dtype0*
_output_shapes
:
Я
2block3_conv2/kernel/Initializer/random_uniform/minConst*
valueB
 *:═╜*&
_class
loc:@block3_conv2/kernel*
dtype0*
_output_shapes
: 
Я
2block3_conv2/kernel/Initializer/random_uniform/maxConst*
valueB
 *:═=*&
_class
loc:@block3_conv2/kernel*
dtype0*
_output_shapes
: 
Д
<block3_conv2/kernel/Initializer/random_uniform/RandomUniformRandomUniform4block3_conv2/kernel/Initializer/random_uniform/shape*

seed *
T0*&
_class
loc:@block3_conv2/kernel*
seed2 *
dtype0*(
_output_shapes
:АА
ъ
2block3_conv2/kernel/Initializer/random_uniform/subSub2block3_conv2/kernel/Initializer/random_uniform/max2block3_conv2/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@block3_conv2/kernel*
_output_shapes
: 
Ж
2block3_conv2/kernel/Initializer/random_uniform/mulMul<block3_conv2/kernel/Initializer/random_uniform/RandomUniform2block3_conv2/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@block3_conv2/kernel*(
_output_shapes
:АА
°
.block3_conv2/kernel/Initializer/random_uniformAdd2block3_conv2/kernel/Initializer/random_uniform/mul2block3_conv2/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@block3_conv2/kernel*(
_output_shapes
:АА
┼
block3_conv2/kernelVarHandleOp*$
shared_nameblock3_conv2/kernel*&
_class
loc:@block3_conv2/kernel*
	container *
shape:АА*
dtype0*
_output_shapes
: 
w
4block3_conv2/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock3_conv2/kernel*
_output_shapes
: 
А
block3_conv2/kernel/AssignAssignVariableOpblock3_conv2/kernel.block3_conv2/kernel/Initializer/random_uniform*
dtype0
Е
'block3_conv2/kernel/Read/ReadVariableOpReadVariableOpblock3_conv2/kernel*
dtype0*(
_output_shapes
:АА
Ш
#block3_conv2/bias/Initializer/zerosConst*
valueBА*    *$
_class
loc:@block3_conv2/bias*
dtype0*
_output_shapes	
:А
▓
block3_conv2/biasVarHandleOp*"
shared_nameblock3_conv2/bias*$
_class
loc:@block3_conv2/bias*
	container *
shape:А*
dtype0*
_output_shapes
: 
s
2block3_conv2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock3_conv2/bias*
_output_shapes
: 
q
block3_conv2/bias/AssignAssignVariableOpblock3_conv2/bias#block3_conv2/bias/Initializer/zeros*
dtype0
t
%block3_conv2/bias/Read/ReadVariableOpReadVariableOpblock3_conv2/bias*
dtype0*
_output_shapes	
:А
k
block3_conv2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
А
"block3_conv2/Conv2D/ReadVariableOpReadVariableOpblock3_conv2/kernel*
dtype0*(
_output_shapes
:АА
Ц
block3_conv2/Conv2DConv2Dblock3_conv1/Relu"block3_conv2/Conv2D/ReadVariableOp*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:         88А
r
#block3_conv2/BiasAdd/ReadVariableOpReadVariableOpblock3_conv2/bias*
dtype0*
_output_shapes	
:А
л
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D#block3_conv2/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*0
_output_shapes
:         88А
j
block3_conv2/ReluRelublock3_conv2/BiasAdd*
T0*0
_output_shapes
:         88А
╡
4block3_conv3/kernel/Initializer/random_uniform/shapeConst*%
valueB"            *&
_class
loc:@block3_conv3/kernel*
dtype0*
_output_shapes
:
Я
2block3_conv3/kernel/Initializer/random_uniform/minConst*
valueB
 *:═╜*&
_class
loc:@block3_conv3/kernel*
dtype0*
_output_shapes
: 
Я
2block3_conv3/kernel/Initializer/random_uniform/maxConst*
valueB
 *:═=*&
_class
loc:@block3_conv3/kernel*
dtype0*
_output_shapes
: 
Д
<block3_conv3/kernel/Initializer/random_uniform/RandomUniformRandomUniform4block3_conv3/kernel/Initializer/random_uniform/shape*

seed *
T0*&
_class
loc:@block3_conv3/kernel*
seed2 *
dtype0*(
_output_shapes
:АА
ъ
2block3_conv3/kernel/Initializer/random_uniform/subSub2block3_conv3/kernel/Initializer/random_uniform/max2block3_conv3/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@block3_conv3/kernel*
_output_shapes
: 
Ж
2block3_conv3/kernel/Initializer/random_uniform/mulMul<block3_conv3/kernel/Initializer/random_uniform/RandomUniform2block3_conv3/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@block3_conv3/kernel*(
_output_shapes
:АА
°
.block3_conv3/kernel/Initializer/random_uniformAdd2block3_conv3/kernel/Initializer/random_uniform/mul2block3_conv3/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@block3_conv3/kernel*(
_output_shapes
:АА
┼
block3_conv3/kernelVarHandleOp*$
shared_nameblock3_conv3/kernel*&
_class
loc:@block3_conv3/kernel*
	container *
shape:АА*
dtype0*
_output_shapes
: 
w
4block3_conv3/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock3_conv3/kernel*
_output_shapes
: 
А
block3_conv3/kernel/AssignAssignVariableOpblock3_conv3/kernel.block3_conv3/kernel/Initializer/random_uniform*
dtype0
Е
'block3_conv3/kernel/Read/ReadVariableOpReadVariableOpblock3_conv3/kernel*
dtype0*(
_output_shapes
:АА
Ш
#block3_conv3/bias/Initializer/zerosConst*
valueBА*    *$
_class
loc:@block3_conv3/bias*
dtype0*
_output_shapes	
:А
▓
block3_conv3/biasVarHandleOp*"
shared_nameblock3_conv3/bias*$
_class
loc:@block3_conv3/bias*
	container *
shape:А*
dtype0*
_output_shapes
: 
s
2block3_conv3/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock3_conv3/bias*
_output_shapes
: 
q
block3_conv3/bias/AssignAssignVariableOpblock3_conv3/bias#block3_conv3/bias/Initializer/zeros*
dtype0
t
%block3_conv3/bias/Read/ReadVariableOpReadVariableOpblock3_conv3/bias*
dtype0*
_output_shapes	
:А
k
block3_conv3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
А
"block3_conv3/Conv2D/ReadVariableOpReadVariableOpblock3_conv3/kernel*
dtype0*(
_output_shapes
:АА
Ц
block3_conv3/Conv2DConv2Dblock3_conv2/Relu"block3_conv3/Conv2D/ReadVariableOp*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:         88А
r
#block3_conv3/BiasAdd/ReadVariableOpReadVariableOpblock3_conv3/bias*
dtype0*
_output_shapes	
:А
л
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D#block3_conv3/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*0
_output_shapes
:         88А
j
block3_conv3/ReluRelublock3_conv3/BiasAdd*
T0*0
_output_shapes
:         88А
┐
block3_pool/MaxPoolMaxPoolblock3_conv3/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*0
_output_shapes
:         А
╡
4block4_conv1/kernel/Initializer/random_uniform/shapeConst*%
valueB"            *&
_class
loc:@block4_conv1/kernel*
dtype0*
_output_shapes
:
Я
2block4_conv1/kernel/Initializer/random_uniform/minConst*
valueB
 *я[ё╝*&
_class
loc:@block4_conv1/kernel*
dtype0*
_output_shapes
: 
Я
2block4_conv1/kernel/Initializer/random_uniform/maxConst*
valueB
 *я[ё<*&
_class
loc:@block4_conv1/kernel*
dtype0*
_output_shapes
: 
Д
<block4_conv1/kernel/Initializer/random_uniform/RandomUniformRandomUniform4block4_conv1/kernel/Initializer/random_uniform/shape*

seed *
T0*&
_class
loc:@block4_conv1/kernel*
seed2 *
dtype0*(
_output_shapes
:АА
ъ
2block4_conv1/kernel/Initializer/random_uniform/subSub2block4_conv1/kernel/Initializer/random_uniform/max2block4_conv1/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@block4_conv1/kernel*
_output_shapes
: 
Ж
2block4_conv1/kernel/Initializer/random_uniform/mulMul<block4_conv1/kernel/Initializer/random_uniform/RandomUniform2block4_conv1/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@block4_conv1/kernel*(
_output_shapes
:АА
°
.block4_conv1/kernel/Initializer/random_uniformAdd2block4_conv1/kernel/Initializer/random_uniform/mul2block4_conv1/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@block4_conv1/kernel*(
_output_shapes
:АА
┼
block4_conv1/kernelVarHandleOp*$
shared_nameblock4_conv1/kernel*&
_class
loc:@block4_conv1/kernel*
	container *
shape:АА*
dtype0*
_output_shapes
: 
w
4block4_conv1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock4_conv1/kernel*
_output_shapes
: 
А
block4_conv1/kernel/AssignAssignVariableOpblock4_conv1/kernel.block4_conv1/kernel/Initializer/random_uniform*
dtype0
Е
'block4_conv1/kernel/Read/ReadVariableOpReadVariableOpblock4_conv1/kernel*
dtype0*(
_output_shapes
:АА
Ш
#block4_conv1/bias/Initializer/zerosConst*
valueBА*    *$
_class
loc:@block4_conv1/bias*
dtype0*
_output_shapes	
:А
▓
block4_conv1/biasVarHandleOp*"
shared_nameblock4_conv1/bias*$
_class
loc:@block4_conv1/bias*
	container *
shape:А*
dtype0*
_output_shapes
: 
s
2block4_conv1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock4_conv1/bias*
_output_shapes
: 
q
block4_conv1/bias/AssignAssignVariableOpblock4_conv1/bias#block4_conv1/bias/Initializer/zeros*
dtype0
t
%block4_conv1/bias/Read/ReadVariableOpReadVariableOpblock4_conv1/bias*
dtype0*
_output_shapes	
:А
k
block4_conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
А
"block4_conv1/Conv2D/ReadVariableOpReadVariableOpblock4_conv1/kernel*
dtype0*(
_output_shapes
:АА
Ш
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool"block4_conv1/Conv2D/ReadVariableOp*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:         А
r
#block4_conv1/BiasAdd/ReadVariableOpReadVariableOpblock4_conv1/bias*
dtype0*
_output_shapes	
:А
л
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D#block4_conv1/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*0
_output_shapes
:         А
j
block4_conv1/ReluRelublock4_conv1/BiasAdd*
T0*0
_output_shapes
:         А
╡
4block4_conv2/kernel/Initializer/random_uniform/shapeConst*%
valueB"            *&
_class
loc:@block4_conv2/kernel*
dtype0*
_output_shapes
:
Я
2block4_conv2/kernel/Initializer/random_uniform/minConst*
valueB
 *ь╤╝*&
_class
loc:@block4_conv2/kernel*
dtype0*
_output_shapes
: 
Я
2block4_conv2/kernel/Initializer/random_uniform/maxConst*
valueB
 *ь╤<*&
_class
loc:@block4_conv2/kernel*
dtype0*
_output_shapes
: 
Д
<block4_conv2/kernel/Initializer/random_uniform/RandomUniformRandomUniform4block4_conv2/kernel/Initializer/random_uniform/shape*

seed *
T0*&
_class
loc:@block4_conv2/kernel*
seed2 *
dtype0*(
_output_shapes
:АА
ъ
2block4_conv2/kernel/Initializer/random_uniform/subSub2block4_conv2/kernel/Initializer/random_uniform/max2block4_conv2/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@block4_conv2/kernel*
_output_shapes
: 
Ж
2block4_conv2/kernel/Initializer/random_uniform/mulMul<block4_conv2/kernel/Initializer/random_uniform/RandomUniform2block4_conv2/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@block4_conv2/kernel*(
_output_shapes
:АА
°
.block4_conv2/kernel/Initializer/random_uniformAdd2block4_conv2/kernel/Initializer/random_uniform/mul2block4_conv2/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@block4_conv2/kernel*(
_output_shapes
:АА
┼
block4_conv2/kernelVarHandleOp*$
shared_nameblock4_conv2/kernel*&
_class
loc:@block4_conv2/kernel*
	container *
shape:АА*
dtype0*
_output_shapes
: 
w
4block4_conv2/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock4_conv2/kernel*
_output_shapes
: 
А
block4_conv2/kernel/AssignAssignVariableOpblock4_conv2/kernel.block4_conv2/kernel/Initializer/random_uniform*
dtype0
Е
'block4_conv2/kernel/Read/ReadVariableOpReadVariableOpblock4_conv2/kernel*
dtype0*(
_output_shapes
:АА
Ш
#block4_conv2/bias/Initializer/zerosConst*
valueBА*    *$
_class
loc:@block4_conv2/bias*
dtype0*
_output_shapes	
:А
▓
block4_conv2/biasVarHandleOp*"
shared_nameblock4_conv2/bias*$
_class
loc:@block4_conv2/bias*
	container *
shape:А*
dtype0*
_output_shapes
: 
s
2block4_conv2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock4_conv2/bias*
_output_shapes
: 
q
block4_conv2/bias/AssignAssignVariableOpblock4_conv2/bias#block4_conv2/bias/Initializer/zeros*
dtype0
t
%block4_conv2/bias/Read/ReadVariableOpReadVariableOpblock4_conv2/bias*
dtype0*
_output_shapes	
:А
k
block4_conv2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
А
"block4_conv2/Conv2D/ReadVariableOpReadVariableOpblock4_conv2/kernel*
dtype0*(
_output_shapes
:АА
Ц
block4_conv2/Conv2DConv2Dblock4_conv1/Relu"block4_conv2/Conv2D/ReadVariableOp*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:         А
r
#block4_conv2/BiasAdd/ReadVariableOpReadVariableOpblock4_conv2/bias*
dtype0*
_output_shapes	
:А
л
block4_conv2/BiasAddBiasAddblock4_conv2/Conv2D#block4_conv2/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*0
_output_shapes
:         А
j
block4_conv2/ReluRelublock4_conv2/BiasAdd*
T0*0
_output_shapes
:         А
╡
4block4_conv3/kernel/Initializer/random_uniform/shapeConst*%
valueB"            *&
_class
loc:@block4_conv3/kernel*
dtype0*
_output_shapes
:
Я
2block4_conv3/kernel/Initializer/random_uniform/minConst*
valueB
 *ь╤╝*&
_class
loc:@block4_conv3/kernel*
dtype0*
_output_shapes
: 
Я
2block4_conv3/kernel/Initializer/random_uniform/maxConst*
valueB
 *ь╤<*&
_class
loc:@block4_conv3/kernel*
dtype0*
_output_shapes
: 
Д
<block4_conv3/kernel/Initializer/random_uniform/RandomUniformRandomUniform4block4_conv3/kernel/Initializer/random_uniform/shape*

seed *
T0*&
_class
loc:@block4_conv3/kernel*
seed2 *
dtype0*(
_output_shapes
:АА
ъ
2block4_conv3/kernel/Initializer/random_uniform/subSub2block4_conv3/kernel/Initializer/random_uniform/max2block4_conv3/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@block4_conv3/kernel*
_output_shapes
: 
Ж
2block4_conv3/kernel/Initializer/random_uniform/mulMul<block4_conv3/kernel/Initializer/random_uniform/RandomUniform2block4_conv3/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@block4_conv3/kernel*(
_output_shapes
:АА
°
.block4_conv3/kernel/Initializer/random_uniformAdd2block4_conv3/kernel/Initializer/random_uniform/mul2block4_conv3/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@block4_conv3/kernel*(
_output_shapes
:АА
┼
block4_conv3/kernelVarHandleOp*$
shared_nameblock4_conv3/kernel*&
_class
loc:@block4_conv3/kernel*
	container *
shape:АА*
dtype0*
_output_shapes
: 
w
4block4_conv3/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock4_conv3/kernel*
_output_shapes
: 
А
block4_conv3/kernel/AssignAssignVariableOpblock4_conv3/kernel.block4_conv3/kernel/Initializer/random_uniform*
dtype0
Е
'block4_conv3/kernel/Read/ReadVariableOpReadVariableOpblock4_conv3/kernel*
dtype0*(
_output_shapes
:АА
Ш
#block4_conv3/bias/Initializer/zerosConst*
valueBА*    *$
_class
loc:@block4_conv3/bias*
dtype0*
_output_shapes	
:А
▓
block4_conv3/biasVarHandleOp*"
shared_nameblock4_conv3/bias*$
_class
loc:@block4_conv3/bias*
	container *
shape:А*
dtype0*
_output_shapes
: 
s
2block4_conv3/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock4_conv3/bias*
_output_shapes
: 
q
block4_conv3/bias/AssignAssignVariableOpblock4_conv3/bias#block4_conv3/bias/Initializer/zeros*
dtype0
t
%block4_conv3/bias/Read/ReadVariableOpReadVariableOpblock4_conv3/bias*
dtype0*
_output_shapes	
:А
k
block4_conv3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
А
"block4_conv3/Conv2D/ReadVariableOpReadVariableOpblock4_conv3/kernel*
dtype0*(
_output_shapes
:АА
Ц
block4_conv3/Conv2DConv2Dblock4_conv2/Relu"block4_conv3/Conv2D/ReadVariableOp*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:         А
r
#block4_conv3/BiasAdd/ReadVariableOpReadVariableOpblock4_conv3/bias*
dtype0*
_output_shapes	
:А
л
block4_conv3/BiasAddBiasAddblock4_conv3/Conv2D#block4_conv3/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*0
_output_shapes
:         А
j
block4_conv3/ReluRelublock4_conv3/BiasAdd*
T0*0
_output_shapes
:         А
┐
block4_pool/MaxPoolMaxPoolblock4_conv3/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*0
_output_shapes
:         А
╡
4block5_conv1/kernel/Initializer/random_uniform/shapeConst*%
valueB"            *&
_class
loc:@block5_conv1/kernel*
dtype0*
_output_shapes
:
Я
2block5_conv1/kernel/Initializer/random_uniform/minConst*
valueB
 *ь╤╝*&
_class
loc:@block5_conv1/kernel*
dtype0*
_output_shapes
: 
Я
2block5_conv1/kernel/Initializer/random_uniform/maxConst*
valueB
 *ь╤<*&
_class
loc:@block5_conv1/kernel*
dtype0*
_output_shapes
: 
Д
<block5_conv1/kernel/Initializer/random_uniform/RandomUniformRandomUniform4block5_conv1/kernel/Initializer/random_uniform/shape*

seed *
T0*&
_class
loc:@block5_conv1/kernel*
seed2 *
dtype0*(
_output_shapes
:АА
ъ
2block5_conv1/kernel/Initializer/random_uniform/subSub2block5_conv1/kernel/Initializer/random_uniform/max2block5_conv1/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@block5_conv1/kernel*
_output_shapes
: 
Ж
2block5_conv1/kernel/Initializer/random_uniform/mulMul<block5_conv1/kernel/Initializer/random_uniform/RandomUniform2block5_conv1/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@block5_conv1/kernel*(
_output_shapes
:АА
°
.block5_conv1/kernel/Initializer/random_uniformAdd2block5_conv1/kernel/Initializer/random_uniform/mul2block5_conv1/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@block5_conv1/kernel*(
_output_shapes
:АА
┼
block5_conv1/kernelVarHandleOp*$
shared_nameblock5_conv1/kernel*&
_class
loc:@block5_conv1/kernel*
	container *
shape:АА*
dtype0*
_output_shapes
: 
w
4block5_conv1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock5_conv1/kernel*
_output_shapes
: 
А
block5_conv1/kernel/AssignAssignVariableOpblock5_conv1/kernel.block5_conv1/kernel/Initializer/random_uniform*
dtype0
Е
'block5_conv1/kernel/Read/ReadVariableOpReadVariableOpblock5_conv1/kernel*
dtype0*(
_output_shapes
:АА
Ш
#block5_conv1/bias/Initializer/zerosConst*
valueBА*    *$
_class
loc:@block5_conv1/bias*
dtype0*
_output_shapes	
:А
▓
block5_conv1/biasVarHandleOp*"
shared_nameblock5_conv1/bias*$
_class
loc:@block5_conv1/bias*
	container *
shape:А*
dtype0*
_output_shapes
: 
s
2block5_conv1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock5_conv1/bias*
_output_shapes
: 
q
block5_conv1/bias/AssignAssignVariableOpblock5_conv1/bias#block5_conv1/bias/Initializer/zeros*
dtype0
t
%block5_conv1/bias/Read/ReadVariableOpReadVariableOpblock5_conv1/bias*
dtype0*
_output_shapes	
:А
k
block5_conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
А
"block5_conv1/Conv2D/ReadVariableOpReadVariableOpblock5_conv1/kernel*
dtype0*(
_output_shapes
:АА
Ш
block5_conv1/Conv2DConv2Dblock4_pool/MaxPool"block5_conv1/Conv2D/ReadVariableOp*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:         А
r
#block5_conv1/BiasAdd/ReadVariableOpReadVariableOpblock5_conv1/bias*
dtype0*
_output_shapes	
:А
л
block5_conv1/BiasAddBiasAddblock5_conv1/Conv2D#block5_conv1/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*0
_output_shapes
:         А
j
block5_conv1/ReluRelublock5_conv1/BiasAdd*
T0*0
_output_shapes
:         А
╡
4block5_conv2/kernel/Initializer/random_uniform/shapeConst*%
valueB"            *&
_class
loc:@block5_conv2/kernel*
dtype0*
_output_shapes
:
Я
2block5_conv2/kernel/Initializer/random_uniform/minConst*
valueB
 *ь╤╝*&
_class
loc:@block5_conv2/kernel*
dtype0*
_output_shapes
: 
Я
2block5_conv2/kernel/Initializer/random_uniform/maxConst*
valueB
 *ь╤<*&
_class
loc:@block5_conv2/kernel*
dtype0*
_output_shapes
: 
Д
<block5_conv2/kernel/Initializer/random_uniform/RandomUniformRandomUniform4block5_conv2/kernel/Initializer/random_uniform/shape*

seed *
T0*&
_class
loc:@block5_conv2/kernel*
seed2 *
dtype0*(
_output_shapes
:АА
ъ
2block5_conv2/kernel/Initializer/random_uniform/subSub2block5_conv2/kernel/Initializer/random_uniform/max2block5_conv2/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@block5_conv2/kernel*
_output_shapes
: 
Ж
2block5_conv2/kernel/Initializer/random_uniform/mulMul<block5_conv2/kernel/Initializer/random_uniform/RandomUniform2block5_conv2/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@block5_conv2/kernel*(
_output_shapes
:АА
°
.block5_conv2/kernel/Initializer/random_uniformAdd2block5_conv2/kernel/Initializer/random_uniform/mul2block5_conv2/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@block5_conv2/kernel*(
_output_shapes
:АА
┼
block5_conv2/kernelVarHandleOp*$
shared_nameblock5_conv2/kernel*&
_class
loc:@block5_conv2/kernel*
	container *
shape:АА*
dtype0*
_output_shapes
: 
w
4block5_conv2/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock5_conv2/kernel*
_output_shapes
: 
А
block5_conv2/kernel/AssignAssignVariableOpblock5_conv2/kernel.block5_conv2/kernel/Initializer/random_uniform*
dtype0
Е
'block5_conv2/kernel/Read/ReadVariableOpReadVariableOpblock5_conv2/kernel*
dtype0*(
_output_shapes
:АА
Ш
#block5_conv2/bias/Initializer/zerosConst*
valueBА*    *$
_class
loc:@block5_conv2/bias*
dtype0*
_output_shapes	
:А
▓
block5_conv2/biasVarHandleOp*"
shared_nameblock5_conv2/bias*$
_class
loc:@block5_conv2/bias*
	container *
shape:А*
dtype0*
_output_shapes
: 
s
2block5_conv2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock5_conv2/bias*
_output_shapes
: 
q
block5_conv2/bias/AssignAssignVariableOpblock5_conv2/bias#block5_conv2/bias/Initializer/zeros*
dtype0
t
%block5_conv2/bias/Read/ReadVariableOpReadVariableOpblock5_conv2/bias*
dtype0*
_output_shapes	
:А
k
block5_conv2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
А
"block5_conv2/Conv2D/ReadVariableOpReadVariableOpblock5_conv2/kernel*
dtype0*(
_output_shapes
:АА
Ц
block5_conv2/Conv2DConv2Dblock5_conv1/Relu"block5_conv2/Conv2D/ReadVariableOp*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:         А
r
#block5_conv2/BiasAdd/ReadVariableOpReadVariableOpblock5_conv2/bias*
dtype0*
_output_shapes	
:А
л
block5_conv2/BiasAddBiasAddblock5_conv2/Conv2D#block5_conv2/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*0
_output_shapes
:         А
j
block5_conv2/ReluRelublock5_conv2/BiasAdd*
T0*0
_output_shapes
:         А
╡
4block5_conv3/kernel/Initializer/random_uniform/shapeConst*%
valueB"            *&
_class
loc:@block5_conv3/kernel*
dtype0*
_output_shapes
:
Я
2block5_conv3/kernel/Initializer/random_uniform/minConst*
valueB
 *ь╤╝*&
_class
loc:@block5_conv3/kernel*
dtype0*
_output_shapes
: 
Я
2block5_conv3/kernel/Initializer/random_uniform/maxConst*
valueB
 *ь╤<*&
_class
loc:@block5_conv3/kernel*
dtype0*
_output_shapes
: 
Д
<block5_conv3/kernel/Initializer/random_uniform/RandomUniformRandomUniform4block5_conv3/kernel/Initializer/random_uniform/shape*

seed *
T0*&
_class
loc:@block5_conv3/kernel*
seed2 *
dtype0*(
_output_shapes
:АА
ъ
2block5_conv3/kernel/Initializer/random_uniform/subSub2block5_conv3/kernel/Initializer/random_uniform/max2block5_conv3/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@block5_conv3/kernel*
_output_shapes
: 
Ж
2block5_conv3/kernel/Initializer/random_uniform/mulMul<block5_conv3/kernel/Initializer/random_uniform/RandomUniform2block5_conv3/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@block5_conv3/kernel*(
_output_shapes
:АА
°
.block5_conv3/kernel/Initializer/random_uniformAdd2block5_conv3/kernel/Initializer/random_uniform/mul2block5_conv3/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@block5_conv3/kernel*(
_output_shapes
:АА
┼
block5_conv3/kernelVarHandleOp*$
shared_nameblock5_conv3/kernel*&
_class
loc:@block5_conv3/kernel*
	container *
shape:АА*
dtype0*
_output_shapes
: 
w
4block5_conv3/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock5_conv3/kernel*
_output_shapes
: 
А
block5_conv3/kernel/AssignAssignVariableOpblock5_conv3/kernel.block5_conv3/kernel/Initializer/random_uniform*
dtype0
Е
'block5_conv3/kernel/Read/ReadVariableOpReadVariableOpblock5_conv3/kernel*
dtype0*(
_output_shapes
:АА
Ш
#block5_conv3/bias/Initializer/zerosConst*
valueBА*    *$
_class
loc:@block5_conv3/bias*
dtype0*
_output_shapes	
:А
▓
block5_conv3/biasVarHandleOp*"
shared_nameblock5_conv3/bias*$
_class
loc:@block5_conv3/bias*
	container *
shape:А*
dtype0*
_output_shapes
: 
s
2block5_conv3/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpblock5_conv3/bias*
_output_shapes
: 
q
block5_conv3/bias/AssignAssignVariableOpblock5_conv3/bias#block5_conv3/bias/Initializer/zeros*
dtype0
t
%block5_conv3/bias/Read/ReadVariableOpReadVariableOpblock5_conv3/bias*
dtype0*
_output_shapes	
:А
k
block5_conv3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
А
"block5_conv3/Conv2D/ReadVariableOpReadVariableOpblock5_conv3/kernel*
dtype0*(
_output_shapes
:АА
Ц
block5_conv3/Conv2DConv2Dblock5_conv2/Relu"block5_conv3/Conv2D/ReadVariableOp*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:         А
r
#block5_conv3/BiasAdd/ReadVariableOpReadVariableOpblock5_conv3/bias*
dtype0*
_output_shapes	
:А
л
block5_conv3/BiasAddBiasAddblock5_conv3/Conv2D#block5_conv3/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*0
_output_shapes
:         А
j
block5_conv3/ReluRelublock5_conv3/BiasAdd*
T0*0
_output_shapes
:         А
┐
block5_pool/MaxPoolMaxPoolblock5_conv3/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*0
_output_shapes
:         А
b
flatten_1/ShapeShapeblock5_pool/MaxPool*
T0*
out_type0*
_output_shapes
:
g
flatten_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
i
flatten_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
i
flatten_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
л
flatten_1/strided_sliceStridedSliceflatten_1/Shapeflatten_1/strided_slice/stackflatten_1/strided_slice/stack_1flatten_1/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
d
flatten_1/Reshape/shape/1Const*
valueB :
         *
dtype0*
_output_shapes
: 
Н
flatten_1/Reshape/shapePackflatten_1/strided_sliceflatten_1/Reshape/shape/1*
T0*

axis *
N*
_output_shapes
:
М
flatten_1/ReshapeReshapeblock5_pool/MaxPoolflatten_1/Reshape/shape*
T0*
Tshape0*)
_output_shapes
:         А─
Ы
+fc1/kernel/Initializer/random_uniform/shapeConst*
valueB" b  А   *
_class
loc:@fc1/kernel*
dtype0*
_output_shapes
:
Н
)fc1/kernel/Initializer/random_uniform/minConst*
valueB
 *√║|╝*
_class
loc:@fc1/kernel*
dtype0*
_output_shapes
: 
Н
)fc1/kernel/Initializer/random_uniform/maxConst*
valueB
 *√║|<*
_class
loc:@fc1/kernel*
dtype0*
_output_shapes
: 
т
3fc1/kernel/Initializer/random_uniform/RandomUniformRandomUniform+fc1/kernel/Initializer/random_uniform/shape*

seed *
T0*
_class
loc:@fc1/kernel*
seed2 *
dtype0*!
_output_shapes
:А─А
╞
)fc1/kernel/Initializer/random_uniform/subSub)fc1/kernel/Initializer/random_uniform/max)fc1/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@fc1/kernel*
_output_shapes
: 
█
)fc1/kernel/Initializer/random_uniform/mulMul3fc1/kernel/Initializer/random_uniform/RandomUniform)fc1/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@fc1/kernel*!
_output_shapes
:А─А
═
%fc1/kernel/Initializer/random_uniformAdd)fc1/kernel/Initializer/random_uniform/mul)fc1/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@fc1/kernel*!
_output_shapes
:А─А
г

fc1/kernelVarHandleOp*
shared_name
fc1/kernel*
_class
loc:@fc1/kernel*
	container *
shape:А─А*
dtype0*
_output_shapes
: 
e
+fc1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp
fc1/kernel*
_output_shapes
: 
e
fc1/kernel/AssignAssignVariableOp
fc1/kernel%fc1/kernel/Initializer/random_uniform*
dtype0
l
fc1/kernel/Read/ReadVariableOpReadVariableOp
fc1/kernel*
dtype0*!
_output_shapes
:А─А
Ж
fc1/bias/Initializer/zerosConst*
valueBА*    *
_class
loc:@fc1/bias*
dtype0*
_output_shapes	
:А
Ч
fc1/biasVarHandleOp*
shared_name
fc1/bias*
_class
loc:@fc1/bias*
	container *
shape:А*
dtype0*
_output_shapes
: 
a
)fc1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpfc1/bias*
_output_shapes
: 
V
fc1/bias/AssignAssignVariableOpfc1/biasfc1/bias/Initializer/zeros*
dtype0
b
fc1/bias/Read/ReadVariableOpReadVariableOpfc1/bias*
dtype0*
_output_shapes	
:А
g
fc1/MatMul/ReadVariableOpReadVariableOp
fc1/kernel*
dtype0*!
_output_shapes
:А─А
Ы

fc1/MatMulMatMulflatten_1/Reshapefc1/MatMul/ReadVariableOp*
transpose_b( *
T0*(
_output_shapes
:         А*
transpose_a( 
`
fc1/BiasAdd/ReadVariableOpReadVariableOpfc1/bias*
dtype0*
_output_shapes	
:А
И
fc1/BiasAddBiasAdd
fc1/MatMulfc1/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*(
_output_shapes
:         А
P
fc1/ReluRelufc1/BiasAdd*
T0*(
_output_shapes
:         А
Ы
+fc2/kernel/Initializer/random_uniform/shapeConst*
valueB"А   А   *
_class
loc:@fc2/kernel*
dtype0*
_output_shapes
:
Н
)fc2/kernel/Initializer/random_uniform/minConst*
valueB
 *q─╛*
_class
loc:@fc2/kernel*
dtype0*
_output_shapes
: 
Н
)fc2/kernel/Initializer/random_uniform/maxConst*
valueB
 *q─>*
_class
loc:@fc2/kernel*
dtype0*
_output_shapes
: 
с
3fc2/kernel/Initializer/random_uniform/RandomUniformRandomUniform+fc2/kernel/Initializer/random_uniform/shape*

seed *
T0*
_class
loc:@fc2/kernel*
seed2 *
dtype0* 
_output_shapes
:
АА
╞
)fc2/kernel/Initializer/random_uniform/subSub)fc2/kernel/Initializer/random_uniform/max)fc2/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@fc2/kernel*
_output_shapes
: 
┌
)fc2/kernel/Initializer/random_uniform/mulMul3fc2/kernel/Initializer/random_uniform/RandomUniform)fc2/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@fc2/kernel* 
_output_shapes
:
АА
╠
%fc2/kernel/Initializer/random_uniformAdd)fc2/kernel/Initializer/random_uniform/mul)fc2/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@fc2/kernel* 
_output_shapes
:
АА
в

fc2/kernelVarHandleOp*
shared_name
fc2/kernel*
_class
loc:@fc2/kernel*
	container *
shape:
АА*
dtype0*
_output_shapes
: 
e
+fc2/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp
fc2/kernel*
_output_shapes
: 
e
fc2/kernel/AssignAssignVariableOp
fc2/kernel%fc2/kernel/Initializer/random_uniform*
dtype0
k
fc2/kernel/Read/ReadVariableOpReadVariableOp
fc2/kernel*
dtype0* 
_output_shapes
:
АА
Ж
fc2/bias/Initializer/zerosConst*
valueBА*    *
_class
loc:@fc2/bias*
dtype0*
_output_shapes	
:А
Ч
fc2/biasVarHandleOp*
shared_name
fc2/bias*
_class
loc:@fc2/bias*
	container *
shape:А*
dtype0*
_output_shapes
: 
a
)fc2/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpfc2/bias*
_output_shapes
: 
V
fc2/bias/AssignAssignVariableOpfc2/biasfc2/bias/Initializer/zeros*
dtype0
b
fc2/bias/Read/ReadVariableOpReadVariableOpfc2/bias*
dtype0*
_output_shapes	
:А
f
fc2/MatMul/ReadVariableOpReadVariableOp
fc2/kernel*
dtype0* 
_output_shapes
:
АА
Т

fc2/MatMulMatMulfc1/Relufc2/MatMul/ReadVariableOp*
transpose_b( *
T0*(
_output_shapes
:         А*
transpose_a( 
`
fc2/BiasAdd/ReadVariableOpReadVariableOpfc2/bias*
dtype0*
_output_shapes	
:А
И
fc2/BiasAddBiasAdd
fc2/MatMulfc2/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*(
_output_shapes
:         А
P
fc2/ReluRelufc2/BiasAdd*
T0*(
_output_shapes
:         А
Э
,fc2a/kernel/Initializer/random_uniform/shapeConst*
valueB"А   А   *
_class
loc:@fc2a/kernel*
dtype0*
_output_shapes
:
П
*fc2a/kernel/Initializer/random_uniform/minConst*
valueB
 *q─╛*
_class
loc:@fc2a/kernel*
dtype0*
_output_shapes
: 
П
*fc2a/kernel/Initializer/random_uniform/maxConst*
valueB
 *q─>*
_class
loc:@fc2a/kernel*
dtype0*
_output_shapes
: 
ф
4fc2a/kernel/Initializer/random_uniform/RandomUniformRandomUniform,fc2a/kernel/Initializer/random_uniform/shape*

seed *
T0*
_class
loc:@fc2a/kernel*
seed2 *
dtype0* 
_output_shapes
:
АА
╩
*fc2a/kernel/Initializer/random_uniform/subSub*fc2a/kernel/Initializer/random_uniform/max*fc2a/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@fc2a/kernel*
_output_shapes
: 
▐
*fc2a/kernel/Initializer/random_uniform/mulMul4fc2a/kernel/Initializer/random_uniform/RandomUniform*fc2a/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@fc2a/kernel* 
_output_shapes
:
АА
╨
&fc2a/kernel/Initializer/random_uniformAdd*fc2a/kernel/Initializer/random_uniform/mul*fc2a/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@fc2a/kernel* 
_output_shapes
:
АА
е
fc2a/kernelVarHandleOp*
shared_namefc2a/kernel*
_class
loc:@fc2a/kernel*
	container *
shape:
АА*
dtype0*
_output_shapes
: 
g
,fc2a/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpfc2a/kernel*
_output_shapes
: 
h
fc2a/kernel/AssignAssignVariableOpfc2a/kernel&fc2a/kernel/Initializer/random_uniform*
dtype0
m
fc2a/kernel/Read/ReadVariableOpReadVariableOpfc2a/kernel*
dtype0* 
_output_shapes
:
АА
И
fc2a/bias/Initializer/zerosConst*
valueBА*    *
_class
loc:@fc2a/bias*
dtype0*
_output_shapes	
:А
Ъ
	fc2a/biasVarHandleOp*
shared_name	fc2a/bias*
_class
loc:@fc2a/bias*
	container *
shape:А*
dtype0*
_output_shapes
: 
c
*fc2a/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp	fc2a/bias*
_output_shapes
: 
Y
fc2a/bias/AssignAssignVariableOp	fc2a/biasfc2a/bias/Initializer/zeros*
dtype0
d
fc2a/bias/Read/ReadVariableOpReadVariableOp	fc2a/bias*
dtype0*
_output_shapes	
:А
h
fc2a/MatMul/ReadVariableOpReadVariableOpfc2a/kernel*
dtype0* 
_output_shapes
:
АА
Ф
fc2a/MatMulMatMulfc2/Relufc2a/MatMul/ReadVariableOp*
transpose_b( *
T0*(
_output_shapes
:         А*
transpose_a( 
b
fc2a/BiasAdd/ReadVariableOpReadVariableOp	fc2a/bias*
dtype0*
_output_shapes	
:А
Л
fc2a/BiasAddBiasAddfc2a/MatMulfc2a/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*(
_output_shapes
:         А
R
	fc2a/ReluRelufc2a/BiasAdd*
T0*(
_output_shapes
:         А
Ы
+fc3/kernel/Initializer/random_uniform/shapeConst*
valueB"А   А   *
_class
loc:@fc3/kernel*
dtype0*
_output_shapes
:
Н
)fc3/kernel/Initializer/random_uniform/minConst*
valueB
 *q─╛*
_class
loc:@fc3/kernel*
dtype0*
_output_shapes
: 
Н
)fc3/kernel/Initializer/random_uniform/maxConst*
valueB
 *q─>*
_class
loc:@fc3/kernel*
dtype0*
_output_shapes
: 
с
3fc3/kernel/Initializer/random_uniform/RandomUniformRandomUniform+fc3/kernel/Initializer/random_uniform/shape*

seed *
T0*
_class
loc:@fc3/kernel*
seed2 *
dtype0* 
_output_shapes
:
АА
╞
)fc3/kernel/Initializer/random_uniform/subSub)fc3/kernel/Initializer/random_uniform/max)fc3/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@fc3/kernel*
_output_shapes
: 
┌
)fc3/kernel/Initializer/random_uniform/mulMul3fc3/kernel/Initializer/random_uniform/RandomUniform)fc3/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@fc3/kernel* 
_output_shapes
:
АА
╠
%fc3/kernel/Initializer/random_uniformAdd)fc3/kernel/Initializer/random_uniform/mul)fc3/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@fc3/kernel* 
_output_shapes
:
АА
в

fc3/kernelVarHandleOp*
shared_name
fc3/kernel*
_class
loc:@fc3/kernel*
	container *
shape:
АА*
dtype0*
_output_shapes
: 
e
+fc3/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp
fc3/kernel*
_output_shapes
: 
e
fc3/kernel/AssignAssignVariableOp
fc3/kernel%fc3/kernel/Initializer/random_uniform*
dtype0
k
fc3/kernel/Read/ReadVariableOpReadVariableOp
fc3/kernel*
dtype0* 
_output_shapes
:
АА
Ж
fc3/bias/Initializer/zerosConst*
valueBА*    *
_class
loc:@fc3/bias*
dtype0*
_output_shapes	
:А
Ч
fc3/biasVarHandleOp*
shared_name
fc3/bias*
_class
loc:@fc3/bias*
	container *
shape:А*
dtype0*
_output_shapes
: 
a
)fc3/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpfc3/bias*
_output_shapes
: 
V
fc3/bias/AssignAssignVariableOpfc3/biasfc3/bias/Initializer/zeros*
dtype0
b
fc3/bias/Read/ReadVariableOpReadVariableOpfc3/bias*
dtype0*
_output_shapes	
:А
f
fc3/MatMul/ReadVariableOpReadVariableOp
fc3/kernel*
dtype0* 
_output_shapes
:
АА
У

fc3/MatMulMatMul	fc2a/Relufc3/MatMul/ReadVariableOp*
transpose_b( *
T0*(
_output_shapes
:         А*
transpose_a( 
`
fc3/BiasAdd/ReadVariableOpReadVariableOpfc3/bias*
dtype0*
_output_shapes	
:А
И
fc3/BiasAddBiasAdd
fc3/MatMulfc3/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*(
_output_shapes
:         А
P
fc3/ReluRelufc3/BiasAdd*
T0*(
_output_shapes
:         А
[
dropout_1/IdentityIdentityfc3/Relu*
T0*(
_output_shapes
:         А
Ы
+fc4/kernel/Initializer/random_uniform/shapeConst*
valueB"А   @   *
_class
loc:@fc4/kernel*
dtype0*
_output_shapes
:
Н
)fc4/kernel/Initializer/random_uniform/minConst*
valueB
 *є5╛*
_class
loc:@fc4/kernel*
dtype0*
_output_shapes
: 
Н
)fc4/kernel/Initializer/random_uniform/maxConst*
valueB
 *є5>*
_class
loc:@fc4/kernel*
dtype0*
_output_shapes
: 
р
3fc4/kernel/Initializer/random_uniform/RandomUniformRandomUniform+fc4/kernel/Initializer/random_uniform/shape*

seed *
T0*
_class
loc:@fc4/kernel*
seed2 *
dtype0*
_output_shapes
:	А@
╞
)fc4/kernel/Initializer/random_uniform/subSub)fc4/kernel/Initializer/random_uniform/max)fc4/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@fc4/kernel*
_output_shapes
: 
┘
)fc4/kernel/Initializer/random_uniform/mulMul3fc4/kernel/Initializer/random_uniform/RandomUniform)fc4/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@fc4/kernel*
_output_shapes
:	А@
╦
%fc4/kernel/Initializer/random_uniformAdd)fc4/kernel/Initializer/random_uniform/mul)fc4/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@fc4/kernel*
_output_shapes
:	А@
б

fc4/kernelVarHandleOp*
shared_name
fc4/kernel*
_class
loc:@fc4/kernel*
	container *
shape:	А@*
dtype0*
_output_shapes
: 
e
+fc4/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp
fc4/kernel*
_output_shapes
: 
e
fc4/kernel/AssignAssignVariableOp
fc4/kernel%fc4/kernel/Initializer/random_uniform*
dtype0
j
fc4/kernel/Read/ReadVariableOpReadVariableOp
fc4/kernel*
dtype0*
_output_shapes
:	А@
Д
fc4/bias/Initializer/zerosConst*
valueB@*    *
_class
loc:@fc4/bias*
dtype0*
_output_shapes
:@
Ц
fc4/biasVarHandleOp*
shared_name
fc4/bias*
_class
loc:@fc4/bias*
	container *
shape:@*
dtype0*
_output_shapes
: 
a
)fc4/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpfc4/bias*
_output_shapes
: 
V
fc4/bias/AssignAssignVariableOpfc4/biasfc4/bias/Initializer/zeros*
dtype0
a
fc4/bias/Read/ReadVariableOpReadVariableOpfc4/bias*
dtype0*
_output_shapes
:@
e
fc4/MatMul/ReadVariableOpReadVariableOp
fc4/kernel*
dtype0*
_output_shapes
:	А@
Ы

fc4/MatMulMatMuldropout_1/Identityfc4/MatMul/ReadVariableOp*
transpose_b( *
T0*'
_output_shapes
:         @*
transpose_a( 
_
fc4/BiasAdd/ReadVariableOpReadVariableOpfc4/bias*
dtype0*
_output_shapes
:@
З
fc4/BiasAddBiasAdd
fc4/MatMulfc4/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*'
_output_shapes
:         @
O
fc4/ReluRelufc4/BiasAdd*
T0*'
_output_shapes
:         @
г
/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"@      *!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
:
Х
-dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *┼└В╛*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
Х
-dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *┼└В>*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
ы
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*

seed *
T0*!
_class
loc:@dense_1/kernel*
seed2 *
dtype0*
_output_shapes

:@
╓
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
: 
ш
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:@
┌
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:@
м
dense_1/kernelVarHandleOp*
shared_namedense_1/kernel*!
_class
loc:@dense_1/kernel*
	container *
shape
:@*
dtype0*
_output_shapes
: 
m
/dense_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/kernel*
_output_shapes
: 
q
dense_1/kernel/AssignAssignVariableOpdense_1/kernel)dense_1/kernel/Initializer/random_uniform*
dtype0
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes

:@
М
dense_1/bias/Initializer/zerosConst*
valueB*    *
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
:
в
dense_1/biasVarHandleOp*
shared_namedense_1/bias*
_class
loc:@dense_1/bias*
	container *
shape:*
dtype0*
_output_shapes
: 
i
-dense_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/bias*
_output_shapes
: 
b
dense_1/bias/AssignAssignVariableOpdense_1/biasdense_1/bias/Initializer/zeros*
dtype0
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes
:
l
dense_1/MatMul/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes

:@
Щ
dense_1/MatMulMatMulfc4/Reludense_1/MatMul/ReadVariableOp*
transpose_b( *
T0*'
_output_shapes
:         *
transpose_a( 
g
dense_1/BiasAdd/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes
:
У
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*'
_output_shapes
:         
]
dense_1/SoftmaxSoftmaxdense_1/BiasAdd*
T0*'
_output_shapes
:         
┤
PlaceholderPlaceholder*?
shape6:4                                    *
dtype0*J
_output_shapes8
6:4                                    
S
AssignVariableOpAssignVariableOpblock1_conv1/kernelPlaceholder*
dtype0
}
ReadVariableOpReadVariableOpblock1_conv1/kernel^AssignVariableOp*
dtype0*&
_output_shapes
:@
h
Placeholder_1Placeholder*
shape:         *
dtype0*#
_output_shapes
:         
U
AssignVariableOp_1AssignVariableOpblock1_conv1/biasPlaceholder_1*
dtype0
s
ReadVariableOp_1ReadVariableOpblock1_conv1/bias^AssignVariableOp_1*
dtype0*
_output_shapes
:@
╢
Placeholder_2Placeholder*?
shape6:4                                    *
dtype0*J
_output_shapes8
6:4                                    
W
AssignVariableOp_2AssignVariableOpblock1_conv2/kernelPlaceholder_2*
dtype0
Б
ReadVariableOp_2ReadVariableOpblock1_conv2/kernel^AssignVariableOp_2*
dtype0*&
_output_shapes
:@@
h
Placeholder_3Placeholder*
shape:         *
dtype0*#
_output_shapes
:         
U
AssignVariableOp_3AssignVariableOpblock1_conv2/biasPlaceholder_3*
dtype0
s
ReadVariableOp_3ReadVariableOpblock1_conv2/bias^AssignVariableOp_3*
dtype0*
_output_shapes
:@
╢
Placeholder_4Placeholder*?
shape6:4                                    *
dtype0*J
_output_shapes8
6:4                                    
W
AssignVariableOp_4AssignVariableOpblock2_conv1/kernelPlaceholder_4*
dtype0
В
ReadVariableOp_4ReadVariableOpblock2_conv1/kernel^AssignVariableOp_4*
dtype0*'
_output_shapes
:@А
h
Placeholder_5Placeholder*
shape:         *
dtype0*#
_output_shapes
:         
U
AssignVariableOp_5AssignVariableOpblock2_conv1/biasPlaceholder_5*
dtype0
t
ReadVariableOp_5ReadVariableOpblock2_conv1/bias^AssignVariableOp_5*
dtype0*
_output_shapes	
:А
╢
Placeholder_6Placeholder*?
shape6:4                                    *
dtype0*J
_output_shapes8
6:4                                    
W
AssignVariableOp_6AssignVariableOpblock2_conv2/kernelPlaceholder_6*
dtype0
Г
ReadVariableOp_6ReadVariableOpblock2_conv2/kernel^AssignVariableOp_6*
dtype0*(
_output_shapes
:АА
h
Placeholder_7Placeholder*
shape:         *
dtype0*#
_output_shapes
:         
U
AssignVariableOp_7AssignVariableOpblock2_conv2/biasPlaceholder_7*
dtype0
t
ReadVariableOp_7ReadVariableOpblock2_conv2/bias^AssignVariableOp_7*
dtype0*
_output_shapes	
:А
╢
Placeholder_8Placeholder*?
shape6:4                                    *
dtype0*J
_output_shapes8
6:4                                    
W
AssignVariableOp_8AssignVariableOpblock3_conv1/kernelPlaceholder_8*
dtype0
Г
ReadVariableOp_8ReadVariableOpblock3_conv1/kernel^AssignVariableOp_8*
dtype0*(
_output_shapes
:АА
h
Placeholder_9Placeholder*
shape:         *
dtype0*#
_output_shapes
:         
U
AssignVariableOp_9AssignVariableOpblock3_conv1/biasPlaceholder_9*
dtype0
t
ReadVariableOp_9ReadVariableOpblock3_conv1/bias^AssignVariableOp_9*
dtype0*
_output_shapes	
:А
╖
Placeholder_10Placeholder*?
shape6:4                                    *
dtype0*J
_output_shapes8
6:4                                    
Y
AssignVariableOp_10AssignVariableOpblock3_conv2/kernelPlaceholder_10*
dtype0
Е
ReadVariableOp_10ReadVariableOpblock3_conv2/kernel^AssignVariableOp_10*
dtype0*(
_output_shapes
:АА
i
Placeholder_11Placeholder*
shape:         *
dtype0*#
_output_shapes
:         
W
AssignVariableOp_11AssignVariableOpblock3_conv2/biasPlaceholder_11*
dtype0
v
ReadVariableOp_11ReadVariableOpblock3_conv2/bias^AssignVariableOp_11*
dtype0*
_output_shapes	
:А
╖
Placeholder_12Placeholder*?
shape6:4                                    *
dtype0*J
_output_shapes8
6:4                                    
Y
AssignVariableOp_12AssignVariableOpblock3_conv3/kernelPlaceholder_12*
dtype0
Е
ReadVariableOp_12ReadVariableOpblock3_conv3/kernel^AssignVariableOp_12*
dtype0*(
_output_shapes
:АА
i
Placeholder_13Placeholder*
shape:         *
dtype0*#
_output_shapes
:         
W
AssignVariableOp_13AssignVariableOpblock3_conv3/biasPlaceholder_13*
dtype0
v
ReadVariableOp_13ReadVariableOpblock3_conv3/bias^AssignVariableOp_13*
dtype0*
_output_shapes	
:А
╖
Placeholder_14Placeholder*?
shape6:4                                    *
dtype0*J
_output_shapes8
6:4                                    
Y
AssignVariableOp_14AssignVariableOpblock4_conv1/kernelPlaceholder_14*
dtype0
Е
ReadVariableOp_14ReadVariableOpblock4_conv1/kernel^AssignVariableOp_14*
dtype0*(
_output_shapes
:АА
i
Placeholder_15Placeholder*
shape:         *
dtype0*#
_output_shapes
:         
W
AssignVariableOp_15AssignVariableOpblock4_conv1/biasPlaceholder_15*
dtype0
v
ReadVariableOp_15ReadVariableOpblock4_conv1/bias^AssignVariableOp_15*
dtype0*
_output_shapes	
:А
╖
Placeholder_16Placeholder*?
shape6:4                                    *
dtype0*J
_output_shapes8
6:4                                    
Y
AssignVariableOp_16AssignVariableOpblock4_conv2/kernelPlaceholder_16*
dtype0
Е
ReadVariableOp_16ReadVariableOpblock4_conv2/kernel^AssignVariableOp_16*
dtype0*(
_output_shapes
:АА
i
Placeholder_17Placeholder*
shape:         *
dtype0*#
_output_shapes
:         
W
AssignVariableOp_17AssignVariableOpblock4_conv2/biasPlaceholder_17*
dtype0
v
ReadVariableOp_17ReadVariableOpblock4_conv2/bias^AssignVariableOp_17*
dtype0*
_output_shapes	
:А
╖
Placeholder_18Placeholder*?
shape6:4                                    *
dtype0*J
_output_shapes8
6:4                                    
Y
AssignVariableOp_18AssignVariableOpblock4_conv3/kernelPlaceholder_18*
dtype0
Е
ReadVariableOp_18ReadVariableOpblock4_conv3/kernel^AssignVariableOp_18*
dtype0*(
_output_shapes
:АА
i
Placeholder_19Placeholder*
shape:         *
dtype0*#
_output_shapes
:         
W
AssignVariableOp_19AssignVariableOpblock4_conv3/biasPlaceholder_19*
dtype0
v
ReadVariableOp_19ReadVariableOpblock4_conv3/bias^AssignVariableOp_19*
dtype0*
_output_shapes	
:А
╖
Placeholder_20Placeholder*?
shape6:4                                    *
dtype0*J
_output_shapes8
6:4                                    
Y
AssignVariableOp_20AssignVariableOpblock5_conv1/kernelPlaceholder_20*
dtype0
Е
ReadVariableOp_20ReadVariableOpblock5_conv1/kernel^AssignVariableOp_20*
dtype0*(
_output_shapes
:АА
i
Placeholder_21Placeholder*
shape:         *
dtype0*#
_output_shapes
:         
W
AssignVariableOp_21AssignVariableOpblock5_conv1/biasPlaceholder_21*
dtype0
v
ReadVariableOp_21ReadVariableOpblock5_conv1/bias^AssignVariableOp_21*
dtype0*
_output_shapes	
:А
╖
Placeholder_22Placeholder*?
shape6:4                                    *
dtype0*J
_output_shapes8
6:4                                    
Y
AssignVariableOp_22AssignVariableOpblock5_conv2/kernelPlaceholder_22*
dtype0
Е
ReadVariableOp_22ReadVariableOpblock5_conv2/kernel^AssignVariableOp_22*
dtype0*(
_output_shapes
:АА
i
Placeholder_23Placeholder*
shape:         *
dtype0*#
_output_shapes
:         
W
AssignVariableOp_23AssignVariableOpblock5_conv2/biasPlaceholder_23*
dtype0
v
ReadVariableOp_23ReadVariableOpblock5_conv2/bias^AssignVariableOp_23*
dtype0*
_output_shapes	
:А
╖
Placeholder_24Placeholder*?
shape6:4                                    *
dtype0*J
_output_shapes8
6:4                                    
Y
AssignVariableOp_24AssignVariableOpblock5_conv3/kernelPlaceholder_24*
dtype0
Е
ReadVariableOp_24ReadVariableOpblock5_conv3/kernel^AssignVariableOp_24*
dtype0*(
_output_shapes
:АА
i
Placeholder_25Placeholder*
shape:         *
dtype0*#
_output_shapes
:         
W
AssignVariableOp_25AssignVariableOpblock5_conv3/biasPlaceholder_25*
dtype0
v
ReadVariableOp_25ReadVariableOpblock5_conv3/bias^AssignVariableOp_25*
dtype0*
_output_shapes	
:А
Г
Placeholder_26Placeholder*%
shape:                  *
dtype0*0
_output_shapes
:                  
P
AssignVariableOp_26AssignVariableOp
fc1/kernelPlaceholder_26*
dtype0
u
ReadVariableOp_26ReadVariableOp
fc1/kernel^AssignVariableOp_26*
dtype0*!
_output_shapes
:А─А
i
Placeholder_27Placeholder*
shape:         *
dtype0*#
_output_shapes
:         
N
AssignVariableOp_27AssignVariableOpfc1/biasPlaceholder_27*
dtype0
m
ReadVariableOp_27ReadVariableOpfc1/bias^AssignVariableOp_27*
dtype0*
_output_shapes	
:А
Г
Placeholder_28Placeholder*%
shape:                  *
dtype0*0
_output_shapes
:                  
P
AssignVariableOp_28AssignVariableOp
fc2/kernelPlaceholder_28*
dtype0
t
ReadVariableOp_28ReadVariableOp
fc2/kernel^AssignVariableOp_28*
dtype0* 
_output_shapes
:
АА
i
Placeholder_29Placeholder*
shape:         *
dtype0*#
_output_shapes
:         
N
AssignVariableOp_29AssignVariableOpfc2/biasPlaceholder_29*
dtype0
m
ReadVariableOp_29ReadVariableOpfc2/bias^AssignVariableOp_29*
dtype0*
_output_shapes	
:А
Г
Placeholder_30Placeholder*%
shape:                  *
dtype0*0
_output_shapes
:                  
Q
AssignVariableOp_30AssignVariableOpfc2a/kernelPlaceholder_30*
dtype0
u
ReadVariableOp_30ReadVariableOpfc2a/kernel^AssignVariableOp_30*
dtype0* 
_output_shapes
:
АА
i
Placeholder_31Placeholder*
shape:         *
dtype0*#
_output_shapes
:         
O
AssignVariableOp_31AssignVariableOp	fc2a/biasPlaceholder_31*
dtype0
n
ReadVariableOp_31ReadVariableOp	fc2a/bias^AssignVariableOp_31*
dtype0*
_output_shapes	
:А
Г
Placeholder_32Placeholder*%
shape:                  *
dtype0*0
_output_shapes
:                  
P
AssignVariableOp_32AssignVariableOp
fc3/kernelPlaceholder_32*
dtype0
t
ReadVariableOp_32ReadVariableOp
fc3/kernel^AssignVariableOp_32*
dtype0* 
_output_shapes
:
АА
i
Placeholder_33Placeholder*
shape:         *
dtype0*#
_output_shapes
:         
N
AssignVariableOp_33AssignVariableOpfc3/biasPlaceholder_33*
dtype0
m
ReadVariableOp_33ReadVariableOpfc3/bias^AssignVariableOp_33*
dtype0*
_output_shapes	
:А
Г
Placeholder_34Placeholder*%
shape:                  *
dtype0*0
_output_shapes
:                  
P
AssignVariableOp_34AssignVariableOp
fc4/kernelPlaceholder_34*
dtype0
s
ReadVariableOp_34ReadVariableOp
fc4/kernel^AssignVariableOp_34*
dtype0*
_output_shapes
:	А@
i
Placeholder_35Placeholder*
shape:         *
dtype0*#
_output_shapes
:         
N
AssignVariableOp_35AssignVariableOpfc4/biasPlaceholder_35*
dtype0
l
ReadVariableOp_35ReadVariableOpfc4/bias^AssignVariableOp_35*
dtype0*
_output_shapes
:@
Г
Placeholder_36Placeholder*%
shape:                  *
dtype0*0
_output_shapes
:                  
T
AssignVariableOp_36AssignVariableOpdense_1/kernelPlaceholder_36*
dtype0
v
ReadVariableOp_36ReadVariableOpdense_1/kernel^AssignVariableOp_36*
dtype0*
_output_shapes

:@
i
Placeholder_37Placeholder*
shape:         *
dtype0*#
_output_shapes
:         
R
AssignVariableOp_37AssignVariableOpdense_1/biasPlaceholder_37*
dtype0
p
ReadVariableOp_37ReadVariableOpdense_1/bias^AssignVariableOp_37*
dtype0*
_output_shapes
:
S
VarIsInitializedOpVarIsInitializedOpblock1_conv2/bias*
_output_shapes
: 
U
VarIsInitializedOp_1VarIsInitializedOpblock4_conv1/bias*
_output_shapes
: 
W
VarIsInitializedOp_2VarIsInitializedOpblock4_conv1/kernel*
_output_shapes
: 
L
VarIsInitializedOp_3VarIsInitializedOpfc2/bias*
_output_shapes
: 
W
VarIsInitializedOp_4VarIsInitializedOpblock4_conv3/kernel*
_output_shapes
: 
W
VarIsInitializedOp_5VarIsInitializedOpblock3_conv2/kernel*
_output_shapes
: 
W
VarIsInitializedOp_6VarIsInitializedOpblock4_conv2/kernel*
_output_shapes
: 
O
VarIsInitializedOp_7VarIsInitializedOpfc2a/kernel*
_output_shapes
: 
N
VarIsInitializedOp_8VarIsInitializedOp
fc4/kernel*
_output_shapes
: 
W
VarIsInitializedOp_9VarIsInitializedOpblock1_conv1/kernel*
_output_shapes
: 
X
VarIsInitializedOp_10VarIsInitializedOpblock1_conv2/kernel*
_output_shapes
: 
V
VarIsInitializedOp_11VarIsInitializedOpblock2_conv2/bias*
_output_shapes
: 
V
VarIsInitializedOp_12VarIsInitializedOpblock3_conv1/bias*
_output_shapes
: 
V
VarIsInitializedOp_13VarIsInitializedOpblock4_conv2/bias*
_output_shapes
: 
X
VarIsInitializedOp_14VarIsInitializedOpblock5_conv2/kernel*
_output_shapes
: 
N
VarIsInitializedOp_15VarIsInitializedOp	fc2a/bias*
_output_shapes
: 
O
VarIsInitializedOp_16VarIsInitializedOp
fc3/kernel*
_output_shapes
: 
Q
VarIsInitializedOp_17VarIsInitializedOpdense_1/bias*
_output_shapes
: 
V
VarIsInitializedOp_18VarIsInitializedOpblock1_conv1/bias*
_output_shapes
: 
V
VarIsInitializedOp_19VarIsInitializedOpblock4_conv3/bias*
_output_shapes
: 
V
VarIsInitializedOp_20VarIsInitializedOpblock5_conv3/bias*
_output_shapes
: 
M
VarIsInitializedOp_21VarIsInitializedOpfc1/bias*
_output_shapes
: 
O
VarIsInitializedOp_22VarIsInitializedOp
fc2/kernel*
_output_shapes
: 
X
VarIsInitializedOp_23VarIsInitializedOpblock3_conv3/kernel*
_output_shapes
: 
V
VarIsInitializedOp_24VarIsInitializedOpblock5_conv1/bias*
_output_shapes
: 
S
VarIsInitializedOp_25VarIsInitializedOpdense_1/kernel*
_output_shapes
: 
X
VarIsInitializedOp_26VarIsInitializedOpblock2_conv2/kernel*
_output_shapes
: 
X
VarIsInitializedOp_27VarIsInitializedOpblock2_conv1/kernel*
_output_shapes
: 
X
VarIsInitializedOp_28VarIsInitializedOpblock3_conv1/kernel*
_output_shapes
: 
X
VarIsInitializedOp_29VarIsInitializedOpblock5_conv3/kernel*
_output_shapes
: 
O
VarIsInitializedOp_30VarIsInitializedOp
fc1/kernel*
_output_shapes
: 
M
VarIsInitializedOp_31VarIsInitializedOpfc3/bias*
_output_shapes
: 
M
VarIsInitializedOp_32VarIsInitializedOpfc4/bias*
_output_shapes
: 
V
VarIsInitializedOp_33VarIsInitializedOpblock2_conv1/bias*
_output_shapes
: 
X
VarIsInitializedOp_34VarIsInitializedOpblock5_conv1/kernel*
_output_shapes
: 
V
VarIsInitializedOp_35VarIsInitializedOpblock5_conv2/bias*
_output_shapes
: 
V
VarIsInitializedOp_36VarIsInitializedOpblock3_conv2/bias*
_output_shapes
: 
V
VarIsInitializedOp_37VarIsInitializedOpblock3_conv3/bias*
_output_shapes
: 
╥
initNoOp^block1_conv1/bias/Assign^block1_conv1/kernel/Assign^block1_conv2/bias/Assign^block1_conv2/kernel/Assign^block2_conv1/bias/Assign^block2_conv1/kernel/Assign^block2_conv2/bias/Assign^block2_conv2/kernel/Assign^block3_conv1/bias/Assign^block3_conv1/kernel/Assign^block3_conv2/bias/Assign^block3_conv2/kernel/Assign^block3_conv3/bias/Assign^block3_conv3/kernel/Assign^block4_conv1/bias/Assign^block4_conv1/kernel/Assign^block4_conv2/bias/Assign^block4_conv2/kernel/Assign^block4_conv3/bias/Assign^block4_conv3/kernel/Assign^block5_conv1/bias/Assign^block5_conv1/kernel/Assign^block5_conv2/bias/Assign^block5_conv2/kernel/Assign^block5_conv3/bias/Assign^block5_conv3/kernel/Assign^dense_1/bias/Assign^dense_1/kernel/Assign^fc1/bias/Assign^fc1/kernel/Assign^fc2/bias/Assign^fc2/kernel/Assign^fc2a/bias/Assign^fc2a/kernel/Assign^fc3/bias/Assign^fc3/kernel/Assign^fc4/bias/Assign^fc4/kernel/Assign
Г
dense_1_targetPlaceholder*%
shape:                  *
dtype0*0
_output_shapes
:                  
v
total/Initializer/zerosConst*
valueB
 *    *
_class

loc:@total*
dtype0*
_output_shapes
: 
Й
totalVarHandleOp*
shared_nametotal*
_class

loc:@total*
	container *
shape: *
dtype0*
_output_shapes
: 
[
&total/IsInitialized/VarIsInitializedOpVarIsInitializedOptotal*
_output_shapes
: 
M
total/AssignAssignVariableOptotaltotal/Initializer/zeros*
dtype0
W
total/Read/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
v
count/Initializer/zerosConst*
valueB
 *    *
_class

loc:@count*
dtype0*
_output_shapes
: 
Й
countVarHandleOp*
shared_namecount*
_class

loc:@count*
	container *
shape: *
dtype0*
_output_shapes
: 
[
&count/IsInitialized/VarIsInitializedOpVarIsInitializedOpcount*
_output_shapes
: 
M
count/AssignAssignVariableOpcountcount/Initializer/zeros*
dtype0
W
count/Read/ReadVariableOpReadVariableOpcount*
dtype0*
_output_shapes
: 
g
metrics/acc/ArgMax/dimensionConst*
valueB :
         *
dtype0*
_output_shapes
: 
Ч
metrics/acc/ArgMaxArgMaxdense_1_targetmetrics/acc/ArgMax/dimension*

Tidx0*
T0*
output_type0	*#
_output_shapes
:         
i
metrics/acc/ArgMax_1/dimensionConst*
valueB :
         *
dtype0*
_output_shapes
: 
Ь
metrics/acc/ArgMax_1ArgMaxdense_1/Softmaxmetrics/acc/ArgMax_1/dimension*

Tidx0*
T0*
output_type0	*#
_output_shapes
:         
Т
metrics/acc/EqualEqualmetrics/acc/ArgMaxmetrics/acc/ArgMax_1*
incompatible_shape_error(*
T0	*#
_output_shapes
:         
x
metrics/acc/CastCastmetrics/acc/Equal*

SrcT0
*
Truncate( *#
_output_shapes
:         *

DstT0
[
metrics/acc/ConstConst*
valueB: *
dtype0*
_output_shapes
:
y
metrics/acc/SumSummetrics/acc/Castmetrics/acc/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
[
metrics/acc/AssignAddVariableOpAssignAddVariableOptotalmetrics/acc/Sum*
dtype0
М
metrics/acc/ReadVariableOpReadVariableOptotal ^metrics/acc/AssignAddVariableOp^metrics/acc/Sum*
dtype0*
_output_shapes
: 
[
metrics/acc/SizeSizemetrics/acc/Cast*
T0*
out_type0*
_output_shapes
: 
l
metrics/acc/Cast_1Castmetrics/acc/Size*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
В
!metrics/acc/AssignAddVariableOp_1AssignAddVariableOpcountmetrics/acc/Cast_1 ^metrics/acc/AssignAddVariableOp*
dtype0
а
metrics/acc/ReadVariableOp_1ReadVariableOpcount ^metrics/acc/AssignAddVariableOp"^metrics/acc/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
З
%metrics/acc/div_no_nan/ReadVariableOpReadVariableOptotal"^metrics/acc/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
Й
'metrics/acc/div_no_nan/ReadVariableOp_1ReadVariableOpcount"^metrics/acc/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
У
metrics/acc/div_no_nanDivNoNan%metrics/acc/div_no_nan/ReadVariableOp'metrics/acc/div_no_nan/ReadVariableOp_1*
T0*
_output_shapes
: 
Y
metrics/acc/IdentityIdentitymetrics/acc/div_no_nan*
T0*
_output_shapes
: 
\
loss/dense_1_loss/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
z
8loss/dense_1_loss/softmax_cross_entropy_with_logits/RankConst*
value	B :*
dtype0*
_output_shapes
: 
И
9loss/dense_1_loss/softmax_cross_entropy_with_logits/ShapeShapedense_1/BiasAdd*
T0*
out_type0*
_output_shapes
:
|
:loss/dense_1_loss/softmax_cross_entropy_with_logits/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
К
;loss/dense_1_loss/softmax_cross_entropy_with_logits/Shape_1Shapedense_1/BiasAdd*
T0*
out_type0*
_output_shapes
:
{
9loss/dense_1_loss/softmax_cross_entropy_with_logits/Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
╓
7loss/dense_1_loss/softmax_cross_entropy_with_logits/SubSub:loss/dense_1_loss/softmax_cross_entropy_with_logits/Rank_19loss/dense_1_loss/softmax_cross_entropy_with_logits/Sub/y*
T0*
_output_shapes
: 
║
?loss/dense_1_loss/softmax_cross_entropy_with_logits/Slice/beginPack7loss/dense_1_loss/softmax_cross_entropy_with_logits/Sub*
T0*

axis *
N*
_output_shapes
:
И
>loss/dense_1_loss/softmax_cross_entropy_with_logits/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
▓
9loss/dense_1_loss/softmax_cross_entropy_with_logits/SliceSlice;loss/dense_1_loss/softmax_cross_entropy_with_logits/Shape_1?loss/dense_1_loss/softmax_cross_entropy_with_logits/Slice/begin>loss/dense_1_loss/softmax_cross_entropy_with_logits/Slice/size*
Index0*
T0*
_output_shapes
:
Ц
Closs/dense_1_loss/softmax_cross_entropy_with_logits/concat/values_0Const*
valueB:
         *
dtype0*
_output_shapes
:
Б
?loss/dense_1_loss/softmax_cross_entropy_with_logits/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
┴
:loss/dense_1_loss/softmax_cross_entropy_with_logits/concatConcatV2Closs/dense_1_loss/softmax_cross_entropy_with_logits/concat/values_09loss/dense_1_loss/softmax_cross_entropy_with_logits/Slice?loss/dense_1_loss/softmax_cross_entropy_with_logits/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
▄
;loss/dense_1_loss/softmax_cross_entropy_with_logits/ReshapeReshapedense_1/BiasAdd:loss/dense_1_loss/softmax_cross_entropy_with_logits/concat*
T0*
Tshape0*0
_output_shapes
:                  
|
:loss/dense_1_loss/softmax_cross_entropy_with_logits/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
Й
;loss/dense_1_loss/softmax_cross_entropy_with_logits/Shape_2Shapedense_1_target*
T0*
out_type0*
_output_shapes
:
}
;loss/dense_1_loss/softmax_cross_entropy_with_logits/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
┌
9loss/dense_1_loss/softmax_cross_entropy_with_logits/Sub_1Sub:loss/dense_1_loss/softmax_cross_entropy_with_logits/Rank_2;loss/dense_1_loss/softmax_cross_entropy_with_logits/Sub_1/y*
T0*
_output_shapes
: 
╛
Aloss/dense_1_loss/softmax_cross_entropy_with_logits/Slice_1/beginPack9loss/dense_1_loss/softmax_cross_entropy_with_logits/Sub_1*
T0*

axis *
N*
_output_shapes
:
К
@loss/dense_1_loss/softmax_cross_entropy_with_logits/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
╕
;loss/dense_1_loss/softmax_cross_entropy_with_logits/Slice_1Slice;loss/dense_1_loss/softmax_cross_entropy_with_logits/Shape_2Aloss/dense_1_loss/softmax_cross_entropy_with_logits/Slice_1/begin@loss/dense_1_loss/softmax_cross_entropy_with_logits/Slice_1/size*
Index0*
T0*
_output_shapes
:
Ш
Eloss/dense_1_loss/softmax_cross_entropy_with_logits/concat_1/values_0Const*
valueB:
         *
dtype0*
_output_shapes
:
Г
Aloss/dense_1_loss/softmax_cross_entropy_with_logits/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
╔
<loss/dense_1_loss/softmax_cross_entropy_with_logits/concat_1ConcatV2Eloss/dense_1_loss/softmax_cross_entropy_with_logits/concat_1/values_0;loss/dense_1_loss/softmax_cross_entropy_with_logits/Slice_1Aloss/dense_1_loss/softmax_cross_entropy_with_logits/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
▀
=loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape_1Reshapedense_1_target<loss/dense_1_loss/softmax_cross_entropy_with_logits/concat_1*
T0*
Tshape0*0
_output_shapes
:                  
Ъ
3loss/dense_1_loss/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits;loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape=loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape_1*
T0*?
_output_shapes-
+:         :                  
}
;loss/dense_1_loss/softmax_cross_entropy_with_logits/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
╪
9loss/dense_1_loss/softmax_cross_entropy_with_logits/Sub_2Sub8loss/dense_1_loss/softmax_cross_entropy_with_logits/Rank;loss/dense_1_loss/softmax_cross_entropy_with_logits/Sub_2/y*
T0*
_output_shapes
: 
Л
Aloss/dense_1_loss/softmax_cross_entropy_with_logits/Slice_2/beginConst*
valueB: *
dtype0*
_output_shapes
:
╜
@loss/dense_1_loss/softmax_cross_entropy_with_logits/Slice_2/sizePack9loss/dense_1_loss/softmax_cross_entropy_with_logits/Sub_2*
T0*

axis *
N*
_output_shapes
:
╢
;loss/dense_1_loss/softmax_cross_entropy_with_logits/Slice_2Slice9loss/dense_1_loss/softmax_cross_entropy_with_logits/ShapeAloss/dense_1_loss/softmax_cross_entropy_with_logits/Slice_2/begin@loss/dense_1_loss/softmax_cross_entropy_with_logits/Slice_2/size*
Index0*
T0*
_output_shapes
:
Ў
=loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape_2Reshape3loss/dense_1_loss/softmax_cross_entropy_with_logits;loss/dense_1_loss/softmax_cross_entropy_with_logits/Slice_2*
T0*
Tshape0*#
_output_shapes
:         
k
&loss/dense_1_loss/weighted_loss/Cast/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Ч
Tloss/dense_1_loss/weighted_loss/broadcast_weights/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
Х
Sloss/dense_1_loss/weighted_loss/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
╨
Sloss/dense_1_loss/weighted_loss/broadcast_weights/assert_broadcastable/values/shapeShape=loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape_2*
T0*
out_type0*
_output_shapes
:
Ф
Rloss/dense_1_loss/weighted_loss/broadcast_weights/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
j
bloss/dense_1_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp
г
Aloss/dense_1_loss/weighted_loss/broadcast_weights/ones_like/ShapeShape=loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape_2c^loss/dense_1_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
ы
Aloss/dense_1_loss/weighted_loss/broadcast_weights/ones_like/ConstConstc^loss/dense_1_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Й
;loss/dense_1_loss/weighted_loss/broadcast_weights/ones_likeFillAloss/dense_1_loss/weighted_loss/broadcast_weights/ones_like/ShapeAloss/dense_1_loss/weighted_loss/broadcast_weights/ones_like/Const*
T0*

index_type0*#
_output_shapes
:         
╦
1loss/dense_1_loss/weighted_loss/broadcast_weightsMul&loss/dense_1_loss/weighted_loss/Cast/x;loss/dense_1_loss/weighted_loss/broadcast_weights/ones_like*
T0*#
_output_shapes
:         
╩
#loss/dense_1_loss/weighted_loss/MulMul=loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape_21loss/dense_1_loss/weighted_loss/broadcast_weights*
T0*#
_output_shapes
:         
c
loss/dense_1_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Ъ
loss/dense_1_loss/SumSum#loss/dense_1_loss/weighted_loss/Mulloss/dense_1_loss/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
|
loss/dense_1_loss/num_elementsSize#loss/dense_1_loss/weighted_loss/Mul*
T0*
out_type0*
_output_shapes
: 
Л
#loss/dense_1_loss/num_elements/CastCastloss/dense_1_loss/num_elements*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
\
loss/dense_1_loss/Const_2Const*
valueB *
dtype0*
_output_shapes
: 
О
loss/dense_1_loss/Sum_1Sumloss/dense_1_loss/Sumloss/dense_1_loss/Const_2*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
В
loss/dense_1_loss/valueDivNoNanloss/dense_1_loss/Sum_1#loss/dense_1_loss/num_elements/Cast*
T0*
_output_shapes
: 
O

loss/mul/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
U
loss/mulMul
loss/mul/xloss/dense_1_loss/value*
T0*
_output_shapes
: 
j
'training/Adam/gradients/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
p
+training/Adam/gradients/gradients/grad_ys_0Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
╖
&training/Adam/gradients/gradients/FillFill'training/Adam/gradients/gradients/Shape+training/Adam/gradients/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
Ь
3training/Adam/gradients/gradients/loss/mul_grad/MulMul&training/Adam/gradients/gradients/Fillloss/dense_1_loss/value*
T0*
_output_shapes
: 
С
5training/Adam/gradients/gradients/loss/mul_grad/Mul_1Mul&training/Adam/gradients/gradients/Fill
loss/mul/x*
T0*
_output_shapes
: 
З
Dtraining/Adam/gradients/gradients/loss/dense_1_loss/value_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Й
Ftraining/Adam/gradients/gradients/loss/dense_1_loss/value_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
╕
Ttraining/Adam/gradients/gradients/loss/dense_1_loss/value_grad/BroadcastGradientArgsBroadcastGradientArgsDtraining/Adam/gradients/gradients/loss/dense_1_loss/value_grad/ShapeFtraining/Adam/gradients/gradients/loss/dense_1_loss/value_grad/Shape_1*
T0*2
_output_shapes 
:         :         
╥
Itraining/Adam/gradients/gradients/loss/dense_1_loss/value_grad/div_no_nanDivNoNan5training/Adam/gradients/gradients/loss/mul_grad/Mul_1#loss/dense_1_loss/num_elements/Cast*
T0*
_output_shapes
: 
и
Btraining/Adam/gradients/gradients/loss/dense_1_loss/value_grad/SumSumItraining/Adam/gradients/gradients/loss/dense_1_loss/value_grad/div_no_nanTtraining/Adam/gradients/gradients/loss/dense_1_loss/value_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
К
Ftraining/Adam/gradients/gradients/loss/dense_1_loss/value_grad/ReshapeReshapeBtraining/Adam/gradients/gradients/loss/dense_1_loss/value_grad/SumDtraining/Adam/gradients/gradients/loss/dense_1_loss/value_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
Г
Btraining/Adam/gradients/gradients/loss/dense_1_loss/value_grad/NegNegloss/dense_1_loss/Sum_1*
T0*
_output_shapes
: 
с
Ktraining/Adam/gradients/gradients/loss/dense_1_loss/value_grad/div_no_nan_1DivNoNanBtraining/Adam/gradients/gradients/loss/dense_1_loss/value_grad/Neg#loss/dense_1_loss/num_elements/Cast*
T0*
_output_shapes
: 
ъ
Ktraining/Adam/gradients/gradients/loss/dense_1_loss/value_grad/div_no_nan_2DivNoNanKtraining/Adam/gradients/gradients/loss/dense_1_loss/value_grad/div_no_nan_1#loss/dense_1_loss/num_elements/Cast*
T0*
_output_shapes
: 
ю
Btraining/Adam/gradients/gradients/loss/dense_1_loss/value_grad/mulMul5training/Adam/gradients/gradients/loss/mul_grad/Mul_1Ktraining/Adam/gradients/gradients/loss/dense_1_loss/value_grad/div_no_nan_2*
T0*
_output_shapes
: 
е
Dtraining/Adam/gradients/gradients/loss/dense_1_loss/value_grad/Sum_1SumBtraining/Adam/gradients/gradients/loss/dense_1_loss/value_grad/mulVtraining/Adam/gradients/gradients/loss/dense_1_loss/value_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
Р
Htraining/Adam/gradients/gradients/loss/dense_1_loss/value_grad/Reshape_1ReshapeDtraining/Adam/gradients/gradients/loss/dense_1_loss/value_grad/Sum_1Ftraining/Adam/gradients/gradients/loss/dense_1_loss/value_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
П
Ltraining/Adam/gradients/gradients/loss/dense_1_loss/Sum_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 
Ц
Ftraining/Adam/gradients/gradients/loss/dense_1_loss/Sum_1_grad/ReshapeReshapeFtraining/Adam/gradients/gradients/loss/dense_1_loss/value_grad/ReshapeLtraining/Adam/gradients/gradients/loss/dense_1_loss/Sum_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
З
Dtraining/Adam/gradients/gradients/loss/dense_1_loss/Sum_1_grad/ConstConst*
valueB *
dtype0*
_output_shapes
: 
М
Ctraining/Adam/gradients/gradients/loss/dense_1_loss/Sum_1_grad/TileTileFtraining/Adam/gradients/gradients/loss/dense_1_loss/Sum_1_grad/ReshapeDtraining/Adam/gradients/gradients/loss/dense_1_loss/Sum_1_grad/Const*

Tmultiples0*
T0*
_output_shapes
: 
Ф
Jtraining/Adam/gradients/gradients/loss/dense_1_loss/Sum_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
У
Dtraining/Adam/gradients/gradients/loss/dense_1_loss/Sum_grad/ReshapeReshapeCtraining/Adam/gradients/gradients/loss/dense_1_loss/Sum_1_grad/TileJtraining/Adam/gradients/gradients/loss/dense_1_loss/Sum_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
е
Btraining/Adam/gradients/gradients/loss/dense_1_loss/Sum_grad/ShapeShape#loss/dense_1_loss/weighted_loss/Mul*
T0*
out_type0*
_output_shapes
:
У
Atraining/Adam/gradients/gradients/loss/dense_1_loss/Sum_grad/TileTileDtraining/Adam/gradients/gradients/loss/dense_1_loss/Sum_grad/ReshapeBtraining/Adam/gradients/gradients/loss/dense_1_loss/Sum_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:         
═
Ptraining/Adam/gradients/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/ShapeShape=loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape_2*
T0*
out_type0*
_output_shapes
:
├
Rtraining/Adam/gradients/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/Shape_1Shape1loss/dense_1_loss/weighted_loss/broadcast_weights*
T0*
out_type0*
_output_shapes
:
▄
`training/Adam/gradients/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgsPtraining/Adam/gradients/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/ShapeRtraining/Adam/gradients/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/Shape_1*
T0*2
_output_shapes 
:         :         
∙
Ntraining/Adam/gradients/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/MulMulAtraining/Adam/gradients/gradients/loss/dense_1_loss/Sum_grad/Tile1loss/dense_1_loss/weighted_loss/broadcast_weights*
T0*#
_output_shapes
:         
╟
Ntraining/Adam/gradients/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/SumSumNtraining/Adam/gradients/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/Mul`training/Adam/gradients/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
╗
Rtraining/Adam/gradients/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/ReshapeReshapeNtraining/Adam/gradients/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/SumPtraining/Adam/gradients/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/Shape*
T0*
Tshape0*#
_output_shapes
:         
З
Ptraining/Adam/gradients/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/Mul_1Mul=loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape_2Atraining/Adam/gradients/gradients/loss/dense_1_loss/Sum_grad/Tile*
T0*#
_output_shapes
:         
═
Ptraining/Adam/gradients/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/Sum_1SumPtraining/Adam/gradients/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/Mul_1btraining/Adam/gradients/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
┴
Ttraining/Adam/gradients/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/Reshape_1ReshapePtraining/Adam/gradients/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/Sum_1Rtraining/Adam/gradients/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:         
▌
jtraining/Adam/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape_2_grad/ShapeShape3loss/dense_1_loss/softmax_cross_entropy_with_logits*
T0*
out_type0*
_output_shapes
:
є
ltraining/Adam/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape_2_grad/ReshapeReshapeRtraining/Adam/gradients/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/Reshapejtraining/Adam/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:         
л
,training/Adam/gradients/gradients/zeros_like	ZerosLike5loss/dense_1_loss/softmax_cross_entropy_with_logits:1*
T0*0
_output_shapes
:                  
┤
itraining/Adam/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits_grad/ExpandDims/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
К
etraining/Adam/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits_grad/ExpandDims
ExpandDimsltraining/Adam/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape_2_grad/Reshapeitraining/Adam/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits_grad/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:         
╛
^training/Adam/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits_grad/mulMuletraining/Adam/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits_grad/ExpandDims5loss/dense_1_loss/softmax_cross_entropy_with_logits:1*
T0*0
_output_shapes
:                  
ы
etraining/Adam/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits_grad/LogSoftmax
LogSoftmax;loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape*
T0*0
_output_shapes
:                  
З
^training/Adam/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits_grad/NegNegetraining/Adam/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits_grad/LogSoftmax*
T0*0
_output_shapes
:                  
╢
ktraining/Adam/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits_grad/ExpandDims_1/dimConst*
valueB :
         *
dtype0*
_output_shapes
: 
О
gtraining/Adam/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits_grad/ExpandDims_1
ExpandDimsltraining/Adam/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape_2_grad/Reshapektraining/Adam/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits_grad/ExpandDims_1/dim*

Tdim0*
T0*'
_output_shapes
:         
ы
`training/Adam/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits_grad/mul_1Mulgtraining/Adam/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits_grad/ExpandDims_1^training/Adam/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits_grad/Neg*
T0*0
_output_shapes
:                  
╖
htraining/Adam/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape_grad/ShapeShapedense_1/BiasAdd*
T0*
out_type0*
_output_shapes
:
 
jtraining/Adam/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape_grad/ReshapeReshape^training/Adam/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits_grad/mulhtraining/Adam/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
∙
Btraining/Adam/gradients/gradients/dense_1/BiasAdd_grad/BiasAddGradBiasAddGradjtraining/Adam/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape_grad/Reshape*
T0*
data_formatNHWC*
_output_shapes
:
й
<training/Adam/gradients/gradients/dense_1/MatMul_grad/MatMulMatMuljtraining/Adam/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape_grad/Reshapedense_1/MatMul/ReadVariableOp*
transpose_b(*
T0*'
_output_shapes
:         @*
transpose_a( 
Н
>training/Adam/gradients/gradients/dense_1/MatMul_grad/MatMul_1MatMulfc4/Relujtraining/Adam/gradients/gradients/loss/dense_1_loss/softmax_cross_entropy_with_logits/Reshape_grad/Reshape*
transpose_b( *
T0*
_output_shapes

:@*
transpose_a(
╛
8training/Adam/gradients/gradients/fc4/Relu_grad/ReluGradReluGrad<training/Adam/gradients/gradients/dense_1/MatMul_grad/MatMulfc4/Relu*
T0*'
_output_shapes
:         @
├
>training/Adam/gradients/gradients/fc4/BiasAdd_grad/BiasAddGradBiasAddGrad8training/Adam/gradients/gradients/fc4/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
Ё
8training/Adam/gradients/gradients/fc4/MatMul_grad/MatMulMatMul8training/Adam/gradients/gradients/fc4/Relu_grad/ReluGradfc4/MatMul/ReadVariableOp*
transpose_b(*
T0*(
_output_shapes
:         А*
transpose_a( 
т
:training/Adam/gradients/gradients/fc4/MatMul_grad/MatMul_1MatMuldropout_1/Identity8training/Adam/gradients/gradients/fc4/Relu_grad/ReluGrad*
transpose_b( *
T0*
_output_shapes
:	А@*
transpose_a(
╗
8training/Adam/gradients/gradients/fc3/Relu_grad/ReluGradReluGrad8training/Adam/gradients/gradients/fc4/MatMul_grad/MatMulfc3/Relu*
T0*(
_output_shapes
:         А
─
>training/Adam/gradients/gradients/fc3/BiasAdd_grad/BiasAddGradBiasAddGrad8training/Adam/gradients/gradients/fc3/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:А
Ё
8training/Adam/gradients/gradients/fc3/MatMul_grad/MatMulMatMul8training/Adam/gradients/gradients/fc3/Relu_grad/ReluGradfc3/MatMul/ReadVariableOp*
transpose_b(*
T0*(
_output_shapes
:         А*
transpose_a( 
┌
:training/Adam/gradients/gradients/fc3/MatMul_grad/MatMul_1MatMul	fc2a/Relu8training/Adam/gradients/gradients/fc3/Relu_grad/ReluGrad*
transpose_b( *
T0* 
_output_shapes
:
АА*
transpose_a(
╜
9training/Adam/gradients/gradients/fc2a/Relu_grad/ReluGradReluGrad8training/Adam/gradients/gradients/fc3/MatMul_grad/MatMul	fc2a/Relu*
T0*(
_output_shapes
:         А
╞
?training/Adam/gradients/gradients/fc2a/BiasAdd_grad/BiasAddGradBiasAddGrad9training/Adam/gradients/gradients/fc2a/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:А
є
9training/Adam/gradients/gradients/fc2a/MatMul_grad/MatMulMatMul9training/Adam/gradients/gradients/fc2a/Relu_grad/ReluGradfc2a/MatMul/ReadVariableOp*
transpose_b(*
T0*(
_output_shapes
:         А*
transpose_a( 
█
;training/Adam/gradients/gradients/fc2a/MatMul_grad/MatMul_1MatMulfc2/Relu9training/Adam/gradients/gradients/fc2a/Relu_grad/ReluGrad*
transpose_b( *
T0* 
_output_shapes
:
АА*
transpose_a(
╝
8training/Adam/gradients/gradients/fc2/Relu_grad/ReluGradReluGrad9training/Adam/gradients/gradients/fc2a/MatMul_grad/MatMulfc2/Relu*
T0*(
_output_shapes
:         А
─
>training/Adam/gradients/gradients/fc2/BiasAdd_grad/BiasAddGradBiasAddGrad8training/Adam/gradients/gradients/fc2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:А
Ё
8training/Adam/gradients/gradients/fc2/MatMul_grad/MatMulMatMul8training/Adam/gradients/gradients/fc2/Relu_grad/ReluGradfc2/MatMul/ReadVariableOp*
transpose_b(*
T0*(
_output_shapes
:         А*
transpose_a( 
┘
:training/Adam/gradients/gradients/fc2/MatMul_grad/MatMul_1MatMulfc1/Relu8training/Adam/gradients/gradients/fc2/Relu_grad/ReluGrad*
transpose_b( *
T0* 
_output_shapes
:
АА*
transpose_a(
╗
8training/Adam/gradients/gradients/fc1/Relu_grad/ReluGradReluGrad8training/Adam/gradients/gradients/fc2/MatMul_grad/MatMulfc1/Relu*
T0*(
_output_shapes
:         А
─
>training/Adam/gradients/gradients/fc1/BiasAdd_grad/BiasAddGradBiasAddGrad8training/Adam/gradients/gradients/fc1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:А
ё
8training/Adam/gradients/gradients/fc1/MatMul_grad/MatMulMatMul8training/Adam/gradients/gradients/fc1/Relu_grad/ReluGradfc1/MatMul/ReadVariableOp*
transpose_b(*
T0*)
_output_shapes
:         А─*
transpose_a( 
у
:training/Adam/gradients/gradients/fc1/MatMul_grad/MatMul_1MatMulflatten_1/Reshape8training/Adam/gradients/gradients/fc1/Relu_grad/ReluGrad*
transpose_b( *
T0*!
_output_shapes
:А─А*
transpose_a(
Н
$training/Adam/iter/Initializer/zerosConst*
value	B	 R *%
_class
loc:@training/Adam/iter*
dtype0	*
_output_shapes
: 
░
training/Adam/iterVarHandleOp*#
shared_nametraining/Adam/iter*%
_class
loc:@training/Adam/iter*
	container *
shape: *
dtype0	*
_output_shapes
: 
u
3training/Adam/iter/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/iter*
_output_shapes
: 
t
training/Adam/iter/AssignAssignVariableOptraining/Adam/iter$training/Adam/iter/Initializer/zeros*
dtype0	
q
&training/Adam/iter/Read/ReadVariableOpReadVariableOptraining/Adam/iter*
dtype0	*
_output_shapes
: 
Ь
.training/Adam/beta_1/Initializer/initial_valueConst*
valueB
 *fff?*'
_class
loc:@training/Adam/beta_1*
dtype0*
_output_shapes
: 
╢
training/Adam/beta_1VarHandleOp*%
shared_nametraining/Adam/beta_1*'
_class
loc:@training/Adam/beta_1*
	container *
shape: *
dtype0*
_output_shapes
: 
y
5training/Adam/beta_1/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/beta_1*
_output_shapes
: 
В
training/Adam/beta_1/AssignAssignVariableOptraining/Adam/beta_1.training/Adam/beta_1/Initializer/initial_value*
dtype0
u
(training/Adam/beta_1/Read/ReadVariableOpReadVariableOptraining/Adam/beta_1*
dtype0*
_output_shapes
: 
Ь
.training/Adam/beta_2/Initializer/initial_valueConst*
valueB
 *w╛?*'
_class
loc:@training/Adam/beta_2*
dtype0*
_output_shapes
: 
╢
training/Adam/beta_2VarHandleOp*%
shared_nametraining/Adam/beta_2*'
_class
loc:@training/Adam/beta_2*
	container *
shape: *
dtype0*
_output_shapes
: 
y
5training/Adam/beta_2/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/beta_2*
_output_shapes
: 
В
training/Adam/beta_2/AssignAssignVariableOptraining/Adam/beta_2.training/Adam/beta_2/Initializer/initial_value*
dtype0
u
(training/Adam/beta_2/Read/ReadVariableOpReadVariableOptraining/Adam/beta_2*
dtype0*
_output_shapes
: 
Ъ
-training/Adam/decay/Initializer/initial_valueConst*
valueB
 *    *&
_class
loc:@training/Adam/decay*
dtype0*
_output_shapes
: 
│
training/Adam/decayVarHandleOp*$
shared_nametraining/Adam/decay*&
_class
loc:@training/Adam/decay*
	container *
shape: *
dtype0*
_output_shapes
: 
w
4training/Adam/decay/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/decay*
_output_shapes
: 

training/Adam/decay/AssignAssignVariableOptraining/Adam/decay-training/Adam/decay/Initializer/initial_value*
dtype0
s
'training/Adam/decay/Read/ReadVariableOpReadVariableOptraining/Adam/decay*
dtype0*
_output_shapes
: 
к
5training/Adam/learning_rate/Initializer/initial_valueConst*
valueB
 *oГ:*.
_class$
" loc:@training/Adam/learning_rate*
dtype0*
_output_shapes
: 
╦
training/Adam/learning_rateVarHandleOp*,
shared_nametraining/Adam/learning_rate*.
_class$
" loc:@training/Adam/learning_rate*
	container *
shape: *
dtype0*
_output_shapes
: 
З
<training/Adam/learning_rate/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/learning_rate*
_output_shapes
: 
Ч
"training/Adam/learning_rate/AssignAssignVariableOptraining/Adam/learning_rate5training/Adam/learning_rate/Initializer/initial_value*
dtype0
Г
/training/Adam/learning_rate/Read/ReadVariableOpReadVariableOptraining/Adam/learning_rate*
dtype0*
_output_shapes
: 
м
<training/Adam/fc1/kernel/m/Initializer/zeros/shape_as_tensorConst*
_class
loc:@fc1/kernel*
valueB" b  А   *
dtype0*
_output_shapes
:
Ц
2training/Adam/fc1/kernel/m/Initializer/zeros/ConstConst*
_class
loc:@fc1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Г
,training/Adam/fc1/kernel/m/Initializer/zerosFill<training/Adam/fc1/kernel/m/Initializer/zeros/shape_as_tensor2training/Adam/fc1/kernel/m/Initializer/zeros/Const*
T0*
_class
loc:@fc1/kernel*

index_type0*!
_output_shapes
:А─А
├
training/Adam/fc1/kernel/mVarHandleOp*+
shared_nametraining/Adam/fc1/kernel/m*
_class
loc:@fc1/kernel*
	container *
shape:А─А*
dtype0*
_output_shapes
: 
д
;training/Adam/fc1/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/fc1/kernel/m*
_class
loc:@fc1/kernel*
_output_shapes
: 
М
!training/Adam/fc1/kernel/m/AssignAssignVariableOptraining/Adam/fc1/kernel/m,training/Adam/fc1/kernel/m/Initializer/zeros*
dtype0
л
.training/Adam/fc1/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/fc1/kernel/m*
_class
loc:@fc1/kernel*
dtype0*!
_output_shapes
:А─А
Ц
*training/Adam/fc1/bias/m/Initializer/zerosConst*
_class
loc:@fc1/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
╖
training/Adam/fc1/bias/mVarHandleOp*)
shared_nametraining/Adam/fc1/bias/m*
_class
loc:@fc1/bias*
	container *
shape:А*
dtype0*
_output_shapes
: 
Ю
9training/Adam/fc1/bias/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/fc1/bias/m*
_class
loc:@fc1/bias*
_output_shapes
: 
Ж
training/Adam/fc1/bias/m/AssignAssignVariableOptraining/Adam/fc1/bias/m*training/Adam/fc1/bias/m/Initializer/zeros*
dtype0
Я
,training/Adam/fc1/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/fc1/bias/m*
_class
loc:@fc1/bias*
dtype0*
_output_shapes	
:А
м
<training/Adam/fc2/kernel/m/Initializer/zeros/shape_as_tensorConst*
_class
loc:@fc2/kernel*
valueB"А   А   *
dtype0*
_output_shapes
:
Ц
2training/Adam/fc2/kernel/m/Initializer/zeros/ConstConst*
_class
loc:@fc2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
В
,training/Adam/fc2/kernel/m/Initializer/zerosFill<training/Adam/fc2/kernel/m/Initializer/zeros/shape_as_tensor2training/Adam/fc2/kernel/m/Initializer/zeros/Const*
T0*
_class
loc:@fc2/kernel*

index_type0* 
_output_shapes
:
АА
┬
training/Adam/fc2/kernel/mVarHandleOp*+
shared_nametraining/Adam/fc2/kernel/m*
_class
loc:@fc2/kernel*
	container *
shape:
АА*
dtype0*
_output_shapes
: 
д
;training/Adam/fc2/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/fc2/kernel/m*
_class
loc:@fc2/kernel*
_output_shapes
: 
М
!training/Adam/fc2/kernel/m/AssignAssignVariableOptraining/Adam/fc2/kernel/m,training/Adam/fc2/kernel/m/Initializer/zeros*
dtype0
к
.training/Adam/fc2/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/fc2/kernel/m*
_class
loc:@fc2/kernel*
dtype0* 
_output_shapes
:
АА
Ц
*training/Adam/fc2/bias/m/Initializer/zerosConst*
_class
loc:@fc2/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
╖
training/Adam/fc2/bias/mVarHandleOp*)
shared_nametraining/Adam/fc2/bias/m*
_class
loc:@fc2/bias*
	container *
shape:А*
dtype0*
_output_shapes
: 
Ю
9training/Adam/fc2/bias/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/fc2/bias/m*
_class
loc:@fc2/bias*
_output_shapes
: 
Ж
training/Adam/fc2/bias/m/AssignAssignVariableOptraining/Adam/fc2/bias/m*training/Adam/fc2/bias/m/Initializer/zeros*
dtype0
Я
,training/Adam/fc2/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/fc2/bias/m*
_class
loc:@fc2/bias*
dtype0*
_output_shapes	
:А
о
=training/Adam/fc2a/kernel/m/Initializer/zeros/shape_as_tensorConst*
_class
loc:@fc2a/kernel*
valueB"А   А   *
dtype0*
_output_shapes
:
Ш
3training/Adam/fc2a/kernel/m/Initializer/zeros/ConstConst*
_class
loc:@fc2a/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Ж
-training/Adam/fc2a/kernel/m/Initializer/zerosFill=training/Adam/fc2a/kernel/m/Initializer/zeros/shape_as_tensor3training/Adam/fc2a/kernel/m/Initializer/zeros/Const*
T0*
_class
loc:@fc2a/kernel*

index_type0* 
_output_shapes
:
АА
┼
training/Adam/fc2a/kernel/mVarHandleOp*,
shared_nametraining/Adam/fc2a/kernel/m*
_class
loc:@fc2a/kernel*
	container *
shape:
АА*
dtype0*
_output_shapes
: 
з
<training/Adam/fc2a/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/fc2a/kernel/m*
_class
loc:@fc2a/kernel*
_output_shapes
: 
П
"training/Adam/fc2a/kernel/m/AssignAssignVariableOptraining/Adam/fc2a/kernel/m-training/Adam/fc2a/kernel/m/Initializer/zeros*
dtype0
н
/training/Adam/fc2a/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/fc2a/kernel/m*
_class
loc:@fc2a/kernel*
dtype0* 
_output_shapes
:
АА
Ш
+training/Adam/fc2a/bias/m/Initializer/zerosConst*
_class
loc:@fc2a/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
║
training/Adam/fc2a/bias/mVarHandleOp**
shared_nametraining/Adam/fc2a/bias/m*
_class
loc:@fc2a/bias*
	container *
shape:А*
dtype0*
_output_shapes
: 
б
:training/Adam/fc2a/bias/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/fc2a/bias/m*
_class
loc:@fc2a/bias*
_output_shapes
: 
Й
 training/Adam/fc2a/bias/m/AssignAssignVariableOptraining/Adam/fc2a/bias/m+training/Adam/fc2a/bias/m/Initializer/zeros*
dtype0
в
-training/Adam/fc2a/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/fc2a/bias/m*
_class
loc:@fc2a/bias*
dtype0*
_output_shapes	
:А
м
<training/Adam/fc3/kernel/m/Initializer/zeros/shape_as_tensorConst*
_class
loc:@fc3/kernel*
valueB"А   А   *
dtype0*
_output_shapes
:
Ц
2training/Adam/fc3/kernel/m/Initializer/zeros/ConstConst*
_class
loc:@fc3/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
В
,training/Adam/fc3/kernel/m/Initializer/zerosFill<training/Adam/fc3/kernel/m/Initializer/zeros/shape_as_tensor2training/Adam/fc3/kernel/m/Initializer/zeros/Const*
T0*
_class
loc:@fc3/kernel*

index_type0* 
_output_shapes
:
АА
┬
training/Adam/fc3/kernel/mVarHandleOp*+
shared_nametraining/Adam/fc3/kernel/m*
_class
loc:@fc3/kernel*
	container *
shape:
АА*
dtype0*
_output_shapes
: 
д
;training/Adam/fc3/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/fc3/kernel/m*
_class
loc:@fc3/kernel*
_output_shapes
: 
М
!training/Adam/fc3/kernel/m/AssignAssignVariableOptraining/Adam/fc3/kernel/m,training/Adam/fc3/kernel/m/Initializer/zeros*
dtype0
к
.training/Adam/fc3/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/fc3/kernel/m*
_class
loc:@fc3/kernel*
dtype0* 
_output_shapes
:
АА
Ц
*training/Adam/fc3/bias/m/Initializer/zerosConst*
_class
loc:@fc3/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
╖
training/Adam/fc3/bias/mVarHandleOp*)
shared_nametraining/Adam/fc3/bias/m*
_class
loc:@fc3/bias*
	container *
shape:А*
dtype0*
_output_shapes
: 
Ю
9training/Adam/fc3/bias/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/fc3/bias/m*
_class
loc:@fc3/bias*
_output_shapes
: 
Ж
training/Adam/fc3/bias/m/AssignAssignVariableOptraining/Adam/fc3/bias/m*training/Adam/fc3/bias/m/Initializer/zeros*
dtype0
Я
,training/Adam/fc3/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/fc3/bias/m*
_class
loc:@fc3/bias*
dtype0*
_output_shapes	
:А
м
<training/Adam/fc4/kernel/m/Initializer/zeros/shape_as_tensorConst*
_class
loc:@fc4/kernel*
valueB"А   @   *
dtype0*
_output_shapes
:
Ц
2training/Adam/fc4/kernel/m/Initializer/zeros/ConstConst*
_class
loc:@fc4/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Б
,training/Adam/fc4/kernel/m/Initializer/zerosFill<training/Adam/fc4/kernel/m/Initializer/zeros/shape_as_tensor2training/Adam/fc4/kernel/m/Initializer/zeros/Const*
T0*
_class
loc:@fc4/kernel*

index_type0*
_output_shapes
:	А@
┴
training/Adam/fc4/kernel/mVarHandleOp*+
shared_nametraining/Adam/fc4/kernel/m*
_class
loc:@fc4/kernel*
	container *
shape:	А@*
dtype0*
_output_shapes
: 
д
;training/Adam/fc4/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/fc4/kernel/m*
_class
loc:@fc4/kernel*
_output_shapes
: 
М
!training/Adam/fc4/kernel/m/AssignAssignVariableOptraining/Adam/fc4/kernel/m,training/Adam/fc4/kernel/m/Initializer/zeros*
dtype0
й
.training/Adam/fc4/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/fc4/kernel/m*
_class
loc:@fc4/kernel*
dtype0*
_output_shapes
:	А@
Ф
*training/Adam/fc4/bias/m/Initializer/zerosConst*
_class
loc:@fc4/bias*
valueB@*    *
dtype0*
_output_shapes
:@
╢
training/Adam/fc4/bias/mVarHandleOp*)
shared_nametraining/Adam/fc4/bias/m*
_class
loc:@fc4/bias*
	container *
shape:@*
dtype0*
_output_shapes
: 
Ю
9training/Adam/fc4/bias/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/fc4/bias/m*
_class
loc:@fc4/bias*
_output_shapes
: 
Ж
training/Adam/fc4/bias/m/AssignAssignVariableOptraining/Adam/fc4/bias/m*training/Adam/fc4/bias/m/Initializer/zeros*
dtype0
Ю
,training/Adam/fc4/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/fc4/bias/m*
_class
loc:@fc4/bias*
dtype0*
_output_shapes
:@
┤
@training/Adam/dense_1/kernel/m/Initializer/zeros/shape_as_tensorConst*!
_class
loc:@dense_1/kernel*
valueB"@      *
dtype0*
_output_shapes
:
Ю
6training/Adam/dense_1/kernel/m/Initializer/zeros/ConstConst*!
_class
loc:@dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Р
0training/Adam/dense_1/kernel/m/Initializer/zerosFill@training/Adam/dense_1/kernel/m/Initializer/zeros/shape_as_tensor6training/Adam/dense_1/kernel/m/Initializer/zeros/Const*
T0*!
_class
loc:@dense_1/kernel*

index_type0*
_output_shapes

:@
╠
training/Adam/dense_1/kernel/mVarHandleOp*/
shared_name training/Adam/dense_1/kernel/m*!
_class
loc:@dense_1/kernel*
	container *
shape
:@*
dtype0*
_output_shapes
: 
░
?training/Adam/dense_1/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_1/kernel/m*!
_class
loc:@dense_1/kernel*
_output_shapes
: 
Ш
%training/Adam/dense_1/kernel/m/AssignAssignVariableOptraining/Adam/dense_1/kernel/m0training/Adam/dense_1/kernel/m/Initializer/zeros*
dtype0
┤
2training/Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_1/kernel/m*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes

:@
Ь
.training/Adam/dense_1/bias/m/Initializer/zerosConst*
_class
loc:@dense_1/bias*
valueB*    *
dtype0*
_output_shapes
:
┬
training/Adam/dense_1/bias/mVarHandleOp*-
shared_nametraining/Adam/dense_1/bias/m*
_class
loc:@dense_1/bias*
	container *
shape:*
dtype0*
_output_shapes
: 
к
=training/Adam/dense_1/bias/m/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_1/bias/m*
_class
loc:@dense_1/bias*
_output_shapes
: 
Т
#training/Adam/dense_1/bias/m/AssignAssignVariableOptraining/Adam/dense_1/bias/m.training/Adam/dense_1/bias/m/Initializer/zeros*
dtype0
к
0training/Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_1/bias/m*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
:
м
<training/Adam/fc1/kernel/v/Initializer/zeros/shape_as_tensorConst*
_class
loc:@fc1/kernel*
valueB" b  А   *
dtype0*
_output_shapes
:
Ц
2training/Adam/fc1/kernel/v/Initializer/zeros/ConstConst*
_class
loc:@fc1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Г
,training/Adam/fc1/kernel/v/Initializer/zerosFill<training/Adam/fc1/kernel/v/Initializer/zeros/shape_as_tensor2training/Adam/fc1/kernel/v/Initializer/zeros/Const*
T0*
_class
loc:@fc1/kernel*

index_type0*!
_output_shapes
:А─А
├
training/Adam/fc1/kernel/vVarHandleOp*+
shared_nametraining/Adam/fc1/kernel/v*
_class
loc:@fc1/kernel*
	container *
shape:А─А*
dtype0*
_output_shapes
: 
д
;training/Adam/fc1/kernel/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/fc1/kernel/v*
_class
loc:@fc1/kernel*
_output_shapes
: 
М
!training/Adam/fc1/kernel/v/AssignAssignVariableOptraining/Adam/fc1/kernel/v,training/Adam/fc1/kernel/v/Initializer/zeros*
dtype0
л
.training/Adam/fc1/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/fc1/kernel/v*
_class
loc:@fc1/kernel*
dtype0*!
_output_shapes
:А─А
Ц
*training/Adam/fc1/bias/v/Initializer/zerosConst*
_class
loc:@fc1/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
╖
training/Adam/fc1/bias/vVarHandleOp*)
shared_nametraining/Adam/fc1/bias/v*
_class
loc:@fc1/bias*
	container *
shape:А*
dtype0*
_output_shapes
: 
Ю
9training/Adam/fc1/bias/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/fc1/bias/v*
_class
loc:@fc1/bias*
_output_shapes
: 
Ж
training/Adam/fc1/bias/v/AssignAssignVariableOptraining/Adam/fc1/bias/v*training/Adam/fc1/bias/v/Initializer/zeros*
dtype0
Я
,training/Adam/fc1/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/fc1/bias/v*
_class
loc:@fc1/bias*
dtype0*
_output_shapes	
:А
м
<training/Adam/fc2/kernel/v/Initializer/zeros/shape_as_tensorConst*
_class
loc:@fc2/kernel*
valueB"А   А   *
dtype0*
_output_shapes
:
Ц
2training/Adam/fc2/kernel/v/Initializer/zeros/ConstConst*
_class
loc:@fc2/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
В
,training/Adam/fc2/kernel/v/Initializer/zerosFill<training/Adam/fc2/kernel/v/Initializer/zeros/shape_as_tensor2training/Adam/fc2/kernel/v/Initializer/zeros/Const*
T0*
_class
loc:@fc2/kernel*

index_type0* 
_output_shapes
:
АА
┬
training/Adam/fc2/kernel/vVarHandleOp*+
shared_nametraining/Adam/fc2/kernel/v*
_class
loc:@fc2/kernel*
	container *
shape:
АА*
dtype0*
_output_shapes
: 
д
;training/Adam/fc2/kernel/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/fc2/kernel/v*
_class
loc:@fc2/kernel*
_output_shapes
: 
М
!training/Adam/fc2/kernel/v/AssignAssignVariableOptraining/Adam/fc2/kernel/v,training/Adam/fc2/kernel/v/Initializer/zeros*
dtype0
к
.training/Adam/fc2/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/fc2/kernel/v*
_class
loc:@fc2/kernel*
dtype0* 
_output_shapes
:
АА
Ц
*training/Adam/fc2/bias/v/Initializer/zerosConst*
_class
loc:@fc2/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
╖
training/Adam/fc2/bias/vVarHandleOp*)
shared_nametraining/Adam/fc2/bias/v*
_class
loc:@fc2/bias*
	container *
shape:А*
dtype0*
_output_shapes
: 
Ю
9training/Adam/fc2/bias/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/fc2/bias/v*
_class
loc:@fc2/bias*
_output_shapes
: 
Ж
training/Adam/fc2/bias/v/AssignAssignVariableOptraining/Adam/fc2/bias/v*training/Adam/fc2/bias/v/Initializer/zeros*
dtype0
Я
,training/Adam/fc2/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/fc2/bias/v*
_class
loc:@fc2/bias*
dtype0*
_output_shapes	
:А
о
=training/Adam/fc2a/kernel/v/Initializer/zeros/shape_as_tensorConst*
_class
loc:@fc2a/kernel*
valueB"А   А   *
dtype0*
_output_shapes
:
Ш
3training/Adam/fc2a/kernel/v/Initializer/zeros/ConstConst*
_class
loc:@fc2a/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Ж
-training/Adam/fc2a/kernel/v/Initializer/zerosFill=training/Adam/fc2a/kernel/v/Initializer/zeros/shape_as_tensor3training/Adam/fc2a/kernel/v/Initializer/zeros/Const*
T0*
_class
loc:@fc2a/kernel*

index_type0* 
_output_shapes
:
АА
┼
training/Adam/fc2a/kernel/vVarHandleOp*,
shared_nametraining/Adam/fc2a/kernel/v*
_class
loc:@fc2a/kernel*
	container *
shape:
АА*
dtype0*
_output_shapes
: 
з
<training/Adam/fc2a/kernel/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/fc2a/kernel/v*
_class
loc:@fc2a/kernel*
_output_shapes
: 
П
"training/Adam/fc2a/kernel/v/AssignAssignVariableOptraining/Adam/fc2a/kernel/v-training/Adam/fc2a/kernel/v/Initializer/zeros*
dtype0
н
/training/Adam/fc2a/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/fc2a/kernel/v*
_class
loc:@fc2a/kernel*
dtype0* 
_output_shapes
:
АА
Ш
+training/Adam/fc2a/bias/v/Initializer/zerosConst*
_class
loc:@fc2a/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
║
training/Adam/fc2a/bias/vVarHandleOp**
shared_nametraining/Adam/fc2a/bias/v*
_class
loc:@fc2a/bias*
	container *
shape:А*
dtype0*
_output_shapes
: 
б
:training/Adam/fc2a/bias/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/fc2a/bias/v*
_class
loc:@fc2a/bias*
_output_shapes
: 
Й
 training/Adam/fc2a/bias/v/AssignAssignVariableOptraining/Adam/fc2a/bias/v+training/Adam/fc2a/bias/v/Initializer/zeros*
dtype0
в
-training/Adam/fc2a/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/fc2a/bias/v*
_class
loc:@fc2a/bias*
dtype0*
_output_shapes	
:А
м
<training/Adam/fc3/kernel/v/Initializer/zeros/shape_as_tensorConst*
_class
loc:@fc3/kernel*
valueB"А   А   *
dtype0*
_output_shapes
:
Ц
2training/Adam/fc3/kernel/v/Initializer/zeros/ConstConst*
_class
loc:@fc3/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
В
,training/Adam/fc3/kernel/v/Initializer/zerosFill<training/Adam/fc3/kernel/v/Initializer/zeros/shape_as_tensor2training/Adam/fc3/kernel/v/Initializer/zeros/Const*
T0*
_class
loc:@fc3/kernel*

index_type0* 
_output_shapes
:
АА
┬
training/Adam/fc3/kernel/vVarHandleOp*+
shared_nametraining/Adam/fc3/kernel/v*
_class
loc:@fc3/kernel*
	container *
shape:
АА*
dtype0*
_output_shapes
: 
д
;training/Adam/fc3/kernel/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/fc3/kernel/v*
_class
loc:@fc3/kernel*
_output_shapes
: 
М
!training/Adam/fc3/kernel/v/AssignAssignVariableOptraining/Adam/fc3/kernel/v,training/Adam/fc3/kernel/v/Initializer/zeros*
dtype0
к
.training/Adam/fc3/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/fc3/kernel/v*
_class
loc:@fc3/kernel*
dtype0* 
_output_shapes
:
АА
Ц
*training/Adam/fc3/bias/v/Initializer/zerosConst*
_class
loc:@fc3/bias*
valueBА*    *
dtype0*
_output_shapes	
:А
╖
training/Adam/fc3/bias/vVarHandleOp*)
shared_nametraining/Adam/fc3/bias/v*
_class
loc:@fc3/bias*
	container *
shape:А*
dtype0*
_output_shapes
: 
Ю
9training/Adam/fc3/bias/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/fc3/bias/v*
_class
loc:@fc3/bias*
_output_shapes
: 
Ж
training/Adam/fc3/bias/v/AssignAssignVariableOptraining/Adam/fc3/bias/v*training/Adam/fc3/bias/v/Initializer/zeros*
dtype0
Я
,training/Adam/fc3/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/fc3/bias/v*
_class
loc:@fc3/bias*
dtype0*
_output_shapes	
:А
м
<training/Adam/fc4/kernel/v/Initializer/zeros/shape_as_tensorConst*
_class
loc:@fc4/kernel*
valueB"А   @   *
dtype0*
_output_shapes
:
Ц
2training/Adam/fc4/kernel/v/Initializer/zeros/ConstConst*
_class
loc:@fc4/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Б
,training/Adam/fc4/kernel/v/Initializer/zerosFill<training/Adam/fc4/kernel/v/Initializer/zeros/shape_as_tensor2training/Adam/fc4/kernel/v/Initializer/zeros/Const*
T0*
_class
loc:@fc4/kernel*

index_type0*
_output_shapes
:	А@
┴
training/Adam/fc4/kernel/vVarHandleOp*+
shared_nametraining/Adam/fc4/kernel/v*
_class
loc:@fc4/kernel*
	container *
shape:	А@*
dtype0*
_output_shapes
: 
д
;training/Adam/fc4/kernel/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/fc4/kernel/v*
_class
loc:@fc4/kernel*
_output_shapes
: 
М
!training/Adam/fc4/kernel/v/AssignAssignVariableOptraining/Adam/fc4/kernel/v,training/Adam/fc4/kernel/v/Initializer/zeros*
dtype0
й
.training/Adam/fc4/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/fc4/kernel/v*
_class
loc:@fc4/kernel*
dtype0*
_output_shapes
:	А@
Ф
*training/Adam/fc4/bias/v/Initializer/zerosConst*
_class
loc:@fc4/bias*
valueB@*    *
dtype0*
_output_shapes
:@
╢
training/Adam/fc4/bias/vVarHandleOp*)
shared_nametraining/Adam/fc4/bias/v*
_class
loc:@fc4/bias*
	container *
shape:@*
dtype0*
_output_shapes
: 
Ю
9training/Adam/fc4/bias/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/fc4/bias/v*
_class
loc:@fc4/bias*
_output_shapes
: 
Ж
training/Adam/fc4/bias/v/AssignAssignVariableOptraining/Adam/fc4/bias/v*training/Adam/fc4/bias/v/Initializer/zeros*
dtype0
Ю
,training/Adam/fc4/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/fc4/bias/v*
_class
loc:@fc4/bias*
dtype0*
_output_shapes
:@
┤
@training/Adam/dense_1/kernel/v/Initializer/zeros/shape_as_tensorConst*!
_class
loc:@dense_1/kernel*
valueB"@      *
dtype0*
_output_shapes
:
Ю
6training/Adam/dense_1/kernel/v/Initializer/zeros/ConstConst*!
_class
loc:@dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Р
0training/Adam/dense_1/kernel/v/Initializer/zerosFill@training/Adam/dense_1/kernel/v/Initializer/zeros/shape_as_tensor6training/Adam/dense_1/kernel/v/Initializer/zeros/Const*
T0*!
_class
loc:@dense_1/kernel*

index_type0*
_output_shapes

:@
╠
training/Adam/dense_1/kernel/vVarHandleOp*/
shared_name training/Adam/dense_1/kernel/v*!
_class
loc:@dense_1/kernel*
	container *
shape
:@*
dtype0*
_output_shapes
: 
░
?training/Adam/dense_1/kernel/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_1/kernel/v*!
_class
loc:@dense_1/kernel*
_output_shapes
: 
Ш
%training/Adam/dense_1/kernel/v/AssignAssignVariableOptraining/Adam/dense_1/kernel/v0training/Adam/dense_1/kernel/v/Initializer/zeros*
dtype0
┤
2training/Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_1/kernel/v*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes

:@
Ь
.training/Adam/dense_1/bias/v/Initializer/zerosConst*
_class
loc:@dense_1/bias*
valueB*    *
dtype0*
_output_shapes
:
┬
training/Adam/dense_1/bias/vVarHandleOp*-
shared_nametraining/Adam/dense_1/bias/v*
_class
loc:@dense_1/bias*
	container *
shape:*
dtype0*
_output_shapes
: 
к
=training/Adam/dense_1/bias/v/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/Adam/dense_1/bias/v*
_class
loc:@dense_1/bias*
_output_shapes
: 
Т
#training/Adam/dense_1/bias/v/AssignAssignVariableOptraining/Adam/dense_1/bias/v.training/Adam/dense_1/bias/v/Initializer/zeros*
dtype0
к
0training/Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_1/bias/v*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
:
y
%training/Adam/Identity/ReadVariableOpReadVariableOptraining/Adam/learning_rate*
dtype0*
_output_shapes
: 
j
training/Adam/IdentityIdentity%training/Adam/Identity/ReadVariableOp*
T0*
_output_shapes
: 
g
training/Adam/ReadVariableOpReadVariableOptraining/Adam/iter*
dtype0	*
_output_shapes
: 
U
training/Adam/add/yConst*
value	B	 R*
dtype0	*
_output_shapes
: 
n
training/Adam/addAddV2training/Adam/ReadVariableOptraining/Adam/add/y*
T0	*
_output_shapes
: 
m
training/Adam/CastCasttraining/Adam/add*

SrcT0	*
Truncate( *
_output_shapes
: *

DstT0
t
'training/Adam/Identity_1/ReadVariableOpReadVariableOptraining/Adam/beta_1*
dtype0*
_output_shapes
: 
n
training/Adam/Identity_1Identity'training/Adam/Identity_1/ReadVariableOp*
T0*
_output_shapes
: 
t
'training/Adam/Identity_2/ReadVariableOpReadVariableOptraining/Adam/beta_2*
dtype0*
_output_shapes
: 
n
training/Adam/Identity_2Identity'training/Adam/Identity_2/ReadVariableOp*
T0*
_output_shapes
: 
g
training/Adam/PowPowtraining/Adam/Identity_1training/Adam/Cast*
T0*
_output_shapes
: 
i
training/Adam/Pow_1Powtraining/Adam/Identity_2training/Adam/Cast*
T0*
_output_shapes
: 
X
training/Adam/sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
c
training/Adam/subSubtraining/Adam/sub/xtraining/Adam/Pow_1*
T0*
_output_shapes
: 
N
training/Adam/SqrtSqrttraining/Adam/sub*
T0*
_output_shapes
: 
Z
training/Adam/sub_1/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
e
training/Adam/sub_1Subtraining/Adam/sub_1/xtraining/Adam/Pow*
T0*
_output_shapes
: 
j
training/Adam/truedivRealDivtraining/Adam/Sqrttraining/Adam/sub_1*
T0*
_output_shapes
: 
h
training/Adam/mulMultraining/Adam/Identitytraining/Adam/truediv*
T0*
_output_shapes
: 
X
training/Adam/ConstConst*
valueB
 *Х┐╓3*
dtype0*
_output_shapes
: 
Z
training/Adam/sub_2/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
l
training/Adam/sub_2Subtraining/Adam/sub_2/xtraining/Adam/Identity_1*
T0*
_output_shapes
: 
Z
training/Adam/sub_3/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
l
training/Adam/sub_3Subtraining/Adam/sub_3/xtraining/Adam/Identity_2*
T0*
_output_shapes
: 
г
6training/Adam/Adam/update_fc1/kernel/ResourceApplyAdamResourceApplyAdam
fc1/kerneltraining/Adam/fc1/kernel/mtraining/Adam/fc1/kernel/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/Const:training/Adam/gradients/gradients/fc1/MatMul_grad/MatMul_1*
use_locking(*
T0*
_class
loc:@fc1/kernel*
use_nesterov( 
Э
4training/Adam/Adam/update_fc1/bias/ResourceApplyAdamResourceApplyAdamfc1/biastraining/Adam/fc1/bias/mtraining/Adam/fc1/bias/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/Const>training/Adam/gradients/gradients/fc1/BiasAdd_grad/BiasAddGrad*
use_locking(*
T0*
_class
loc:@fc1/bias*
use_nesterov( 
г
6training/Adam/Adam/update_fc2/kernel/ResourceApplyAdamResourceApplyAdam
fc2/kerneltraining/Adam/fc2/kernel/mtraining/Adam/fc2/kernel/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/Const:training/Adam/gradients/gradients/fc2/MatMul_grad/MatMul_1*
use_locking(*
T0*
_class
loc:@fc2/kernel*
use_nesterov( 
Э
4training/Adam/Adam/update_fc2/bias/ResourceApplyAdamResourceApplyAdamfc2/biastraining/Adam/fc2/bias/mtraining/Adam/fc2/bias/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/Const>training/Adam/gradients/gradients/fc2/BiasAdd_grad/BiasAddGrad*
use_locking(*
T0*
_class
loc:@fc2/bias*
use_nesterov( 
й
7training/Adam/Adam/update_fc2a/kernel/ResourceApplyAdamResourceApplyAdamfc2a/kerneltraining/Adam/fc2a/kernel/mtraining/Adam/fc2a/kernel/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/Const;training/Adam/gradients/gradients/fc2a/MatMul_grad/MatMul_1*
use_locking(*
T0*
_class
loc:@fc2a/kernel*
use_nesterov( 
г
5training/Adam/Adam/update_fc2a/bias/ResourceApplyAdamResourceApplyAdam	fc2a/biastraining/Adam/fc2a/bias/mtraining/Adam/fc2a/bias/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/Const?training/Adam/gradients/gradients/fc2a/BiasAdd_grad/BiasAddGrad*
use_locking(*
T0*
_class
loc:@fc2a/bias*
use_nesterov( 
г
6training/Adam/Adam/update_fc3/kernel/ResourceApplyAdamResourceApplyAdam
fc3/kerneltraining/Adam/fc3/kernel/mtraining/Adam/fc3/kernel/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/Const:training/Adam/gradients/gradients/fc3/MatMul_grad/MatMul_1*
use_locking(*
T0*
_class
loc:@fc3/kernel*
use_nesterov( 
Э
4training/Adam/Adam/update_fc3/bias/ResourceApplyAdamResourceApplyAdamfc3/biastraining/Adam/fc3/bias/mtraining/Adam/fc3/bias/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/Const>training/Adam/gradients/gradients/fc3/BiasAdd_grad/BiasAddGrad*
use_locking(*
T0*
_class
loc:@fc3/bias*
use_nesterov( 
г
6training/Adam/Adam/update_fc4/kernel/ResourceApplyAdamResourceApplyAdam
fc4/kerneltraining/Adam/fc4/kernel/mtraining/Adam/fc4/kernel/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/Const:training/Adam/gradients/gradients/fc4/MatMul_grad/MatMul_1*
use_locking(*
T0*
_class
loc:@fc4/kernel*
use_nesterov( 
Э
4training/Adam/Adam/update_fc4/bias/ResourceApplyAdamResourceApplyAdamfc4/biastraining/Adam/fc4/bias/mtraining/Adam/fc4/bias/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/Const>training/Adam/gradients/gradients/fc4/BiasAdd_grad/BiasAddGrad*
use_locking(*
T0*
_class
loc:@fc4/bias*
use_nesterov( 
╗
:training/Adam/Adam/update_dense_1/kernel/ResourceApplyAdamResourceApplyAdamdense_1/kerneltraining/Adam/dense_1/kernel/mtraining/Adam/dense_1/kernel/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/Const>training/Adam/gradients/gradients/dense_1/MatMul_grad/MatMul_1*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
use_nesterov( 
╡
8training/Adam/Adam/update_dense_1/bias/ResourceApplyAdamResourceApplyAdamdense_1/biastraining/Adam/dense_1/bias/mtraining/Adam/dense_1/bias/vtraining/Adam/Powtraining/Adam/Pow_1training/Adam/Identitytraining/Adam/Identity_1training/Adam/Identity_2training/Adam/ConstBtraining/Adam/gradients/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
use_locking(*
T0*
_class
loc:@dense_1/bias*
use_nesterov( 
Д
training/Adam/Adam/ConstConst9^training/Adam/Adam/update_dense_1/bias/ResourceApplyAdam;^training/Adam/Adam/update_dense_1/kernel/ResourceApplyAdam5^training/Adam/Adam/update_fc1/bias/ResourceApplyAdam7^training/Adam/Adam/update_fc1/kernel/ResourceApplyAdam5^training/Adam/Adam/update_fc2/bias/ResourceApplyAdam7^training/Adam/Adam/update_fc2/kernel/ResourceApplyAdam6^training/Adam/Adam/update_fc2a/bias/ResourceApplyAdam8^training/Adam/Adam/update_fc2a/kernel/ResourceApplyAdam5^training/Adam/Adam/update_fc3/bias/ResourceApplyAdam7^training/Adam/Adam/update_fc3/kernel/ResourceApplyAdam5^training/Adam/Adam/update_fc4/bias/ResourceApplyAdam7^training/Adam/Adam/update_fc4/kernel/ResourceApplyAdam*
value	B	 R*
dtype0	*
_output_shapes
: 
x
&training/Adam/Adam/AssignAddVariableOpAssignAddVariableOptraining/Adam/itertraining/Adam/Adam/Const*
dtype0	
┐
!training/Adam/Adam/ReadVariableOpReadVariableOptraining/Adam/iter'^training/Adam/Adam/AssignAddVariableOp9^training/Adam/Adam/update_dense_1/bias/ResourceApplyAdam;^training/Adam/Adam/update_dense_1/kernel/ResourceApplyAdam5^training/Adam/Adam/update_fc1/bias/ResourceApplyAdam7^training/Adam/Adam/update_fc1/kernel/ResourceApplyAdam5^training/Adam/Adam/update_fc2/bias/ResourceApplyAdam7^training/Adam/Adam/update_fc2/kernel/ResourceApplyAdam6^training/Adam/Adam/update_fc2a/bias/ResourceApplyAdam8^training/Adam/Adam/update_fc2a/kernel/ResourceApplyAdam5^training/Adam/Adam/update_fc3/bias/ResourceApplyAdam7^training/Adam/Adam/update_fc3/kernel/ResourceApplyAdam5^training/Adam/Adam/update_fc4/bias/ResourceApplyAdam7^training/Adam/Adam/update_fc4/kernel/ResourceApplyAdam*
dtype0	*
_output_shapes
: 
Q
training_1/group_depsNoOp	^loss/mul'^training/Adam/Adam/AssignAddVariableOp
]
VarIsInitializedOp_38VarIsInitializedOptraining/Adam/fc2/bias/m*
_output_shapes
: 
_
VarIsInitializedOp_39VarIsInitializedOptraining/Adam/fc2/kernel/v*
_output_shapes
: 
]
VarIsInitializedOp_40VarIsInitializedOptraining/Adam/fc2/bias/v*
_output_shapes
: 
c
VarIsInitializedOp_41VarIsInitializedOptraining/Adam/dense_1/kernel/v*
_output_shapes
: 
X
VarIsInitializedOp_42VarIsInitializedOptraining/Adam/decay*
_output_shapes
: 
`
VarIsInitializedOp_43VarIsInitializedOptraining/Adam/fc2a/kernel/m*
_output_shapes
: 
_
VarIsInitializedOp_44VarIsInitializedOptraining/Adam/fc3/kernel/m*
_output_shapes
: 
J
VarIsInitializedOp_45VarIsInitializedOptotal*
_output_shapes
: 
]
VarIsInitializedOp_46VarIsInitializedOptraining/Adam/fc3/bias/m*
_output_shapes
: 
]
VarIsInitializedOp_47VarIsInitializedOptraining/Adam/fc3/bias/v*
_output_shapes
: 
_
VarIsInitializedOp_48VarIsInitializedOptraining/Adam/fc2/kernel/m*
_output_shapes
: 
`
VarIsInitializedOp_49VarIsInitializedOptraining/Adam/learning_rate*
_output_shapes
: 
_
VarIsInitializedOp_50VarIsInitializedOptraining/Adam/fc4/kernel/m*
_output_shapes
: 
^
VarIsInitializedOp_51VarIsInitializedOptraining/Adam/fc2a/bias/v*
_output_shapes
: 
_
VarIsInitializedOp_52VarIsInitializedOptraining/Adam/fc3/kernel/v*
_output_shapes
: 
a
VarIsInitializedOp_53VarIsInitializedOptraining/Adam/dense_1/bias/v*
_output_shapes
: 
]
VarIsInitializedOp_54VarIsInitializedOptraining/Adam/fc1/bias/m*
_output_shapes
: 
c
VarIsInitializedOp_55VarIsInitializedOptraining/Adam/dense_1/kernel/m*
_output_shapes
: 
]
VarIsInitializedOp_56VarIsInitializedOptraining/Adam/fc4/bias/v*
_output_shapes
: 
a
VarIsInitializedOp_57VarIsInitializedOptraining/Adam/dense_1/bias/m*
_output_shapes
: 
Y
VarIsInitializedOp_58VarIsInitializedOptraining/Adam/beta_2*
_output_shapes
: 
_
VarIsInitializedOp_59VarIsInitializedOptraining/Adam/fc1/kernel/m*
_output_shapes
: 
^
VarIsInitializedOp_60VarIsInitializedOptraining/Adam/fc2a/bias/m*
_output_shapes
: 
J
VarIsInitializedOp_61VarIsInitializedOpcount*
_output_shapes
: 
Y
VarIsInitializedOp_62VarIsInitializedOptraining/Adam/beta_1*
_output_shapes
: 
W
VarIsInitializedOp_63VarIsInitializedOptraining/Adam/iter*
_output_shapes
: 
_
VarIsInitializedOp_64VarIsInitializedOptraining/Adam/fc1/kernel/v*
_output_shapes
: 
]
VarIsInitializedOp_65VarIsInitializedOptraining/Adam/fc1/bias/v*
_output_shapes
: 
`
VarIsInitializedOp_66VarIsInitializedOptraining/Adam/fc2a/kernel/v*
_output_shapes
: 
_
VarIsInitializedOp_67VarIsInitializedOptraining/Adam/fc4/kernel/v*
_output_shapes
: 
]
VarIsInitializedOp_68VarIsInitializedOptraining/Adam/fc4/bias/m*
_output_shapes
: 
в
init_1NoOp^count/Assign^total/Assign^training/Adam/beta_1/Assign^training/Adam/beta_2/Assign^training/Adam/decay/Assign$^training/Adam/dense_1/bias/m/Assign$^training/Adam/dense_1/bias/v/Assign&^training/Adam/dense_1/kernel/m/Assign&^training/Adam/dense_1/kernel/v/Assign ^training/Adam/fc1/bias/m/Assign ^training/Adam/fc1/bias/v/Assign"^training/Adam/fc1/kernel/m/Assign"^training/Adam/fc1/kernel/v/Assign ^training/Adam/fc2/bias/m/Assign ^training/Adam/fc2/bias/v/Assign"^training/Adam/fc2/kernel/m/Assign"^training/Adam/fc2/kernel/v/Assign!^training/Adam/fc2a/bias/m/Assign!^training/Adam/fc2a/bias/v/Assign#^training/Adam/fc2a/kernel/m/Assign#^training/Adam/fc2a/kernel/v/Assign ^training/Adam/fc3/bias/m/Assign ^training/Adam/fc3/bias/v/Assign"^training/Adam/fc3/kernel/m/Assign"^training/Adam/fc3/kernel/v/Assign ^training/Adam/fc4/bias/m/Assign ^training/Adam/fc4/bias/v/Assign"^training/Adam/fc4/kernel/m/Assign"^training/Adam/fc4/kernel/v/Assign^training/Adam/iter/Assign#^training/Adam/learning_rate/Assign
O
Placeholder_38Placeholder*
shape: *
dtype0	*
_output_shapes
: 
X
AssignVariableOp_38AssignVariableOptraining/Adam/iterPlaceholder_38*
dtype0	
r
ReadVariableOp_38ReadVariableOptraining/Adam/iter^AssignVariableOp_38*
dtype0	*
_output_shapes
: 
Г
Placeholder_39Placeholder*%
shape:                  *
dtype0*0
_output_shapes
:                  
`
AssignVariableOp_39AssignVariableOptraining/Adam/fc1/kernel/mPlaceholder_39*
dtype0
Е
ReadVariableOp_39ReadVariableOptraining/Adam/fc1/kernel/m^AssignVariableOp_39*
dtype0*!
_output_shapes
:А─А
i
Placeholder_40Placeholder*
shape:         *
dtype0*#
_output_shapes
:         
^
AssignVariableOp_40AssignVariableOptraining/Adam/fc1/bias/mPlaceholder_40*
dtype0
}
ReadVariableOp_40ReadVariableOptraining/Adam/fc1/bias/m^AssignVariableOp_40*
dtype0*
_output_shapes	
:А
Г
Placeholder_41Placeholder*%
shape:                  *
dtype0*0
_output_shapes
:                  
`
AssignVariableOp_41AssignVariableOptraining/Adam/fc2/kernel/mPlaceholder_41*
dtype0
Д
ReadVariableOp_41ReadVariableOptraining/Adam/fc2/kernel/m^AssignVariableOp_41*
dtype0* 
_output_shapes
:
АА
i
Placeholder_42Placeholder*
shape:         *
dtype0*#
_output_shapes
:         
^
AssignVariableOp_42AssignVariableOptraining/Adam/fc2/bias/mPlaceholder_42*
dtype0
}
ReadVariableOp_42ReadVariableOptraining/Adam/fc2/bias/m^AssignVariableOp_42*
dtype0*
_output_shapes	
:А
Г
Placeholder_43Placeholder*%
shape:                  *
dtype0*0
_output_shapes
:                  
a
AssignVariableOp_43AssignVariableOptraining/Adam/fc2a/kernel/mPlaceholder_43*
dtype0
Е
ReadVariableOp_43ReadVariableOptraining/Adam/fc2a/kernel/m^AssignVariableOp_43*
dtype0* 
_output_shapes
:
АА
i
Placeholder_44Placeholder*
shape:         *
dtype0*#
_output_shapes
:         
_
AssignVariableOp_44AssignVariableOptraining/Adam/fc2a/bias/mPlaceholder_44*
dtype0
~
ReadVariableOp_44ReadVariableOptraining/Adam/fc2a/bias/m^AssignVariableOp_44*
dtype0*
_output_shapes	
:А
Г
Placeholder_45Placeholder*%
shape:                  *
dtype0*0
_output_shapes
:                  
`
AssignVariableOp_45AssignVariableOptraining/Adam/fc3/kernel/mPlaceholder_45*
dtype0
Д
ReadVariableOp_45ReadVariableOptraining/Adam/fc3/kernel/m^AssignVariableOp_45*
dtype0* 
_output_shapes
:
АА
i
Placeholder_46Placeholder*
shape:         *
dtype0*#
_output_shapes
:         
^
AssignVariableOp_46AssignVariableOptraining/Adam/fc3/bias/mPlaceholder_46*
dtype0
}
ReadVariableOp_46ReadVariableOptraining/Adam/fc3/bias/m^AssignVariableOp_46*
dtype0*
_output_shapes	
:А
Г
Placeholder_47Placeholder*%
shape:                  *
dtype0*0
_output_shapes
:                  
`
AssignVariableOp_47AssignVariableOptraining/Adam/fc4/kernel/mPlaceholder_47*
dtype0
Г
ReadVariableOp_47ReadVariableOptraining/Adam/fc4/kernel/m^AssignVariableOp_47*
dtype0*
_output_shapes
:	А@
i
Placeholder_48Placeholder*
shape:         *
dtype0*#
_output_shapes
:         
^
AssignVariableOp_48AssignVariableOptraining/Adam/fc4/bias/mPlaceholder_48*
dtype0
|
ReadVariableOp_48ReadVariableOptraining/Adam/fc4/bias/m^AssignVariableOp_48*
dtype0*
_output_shapes
:@
Г
Placeholder_49Placeholder*%
shape:                  *
dtype0*0
_output_shapes
:                  
d
AssignVariableOp_49AssignVariableOptraining/Adam/dense_1/kernel/mPlaceholder_49*
dtype0
Ж
ReadVariableOp_49ReadVariableOptraining/Adam/dense_1/kernel/m^AssignVariableOp_49*
dtype0*
_output_shapes

:@
i
Placeholder_50Placeholder*
shape:         *
dtype0*#
_output_shapes
:         
b
AssignVariableOp_50AssignVariableOptraining/Adam/dense_1/bias/mPlaceholder_50*
dtype0
А
ReadVariableOp_50ReadVariableOptraining/Adam/dense_1/bias/m^AssignVariableOp_50*
dtype0*
_output_shapes
:
Г
Placeholder_51Placeholder*%
shape:                  *
dtype0*0
_output_shapes
:                  
`
AssignVariableOp_51AssignVariableOptraining/Adam/fc1/kernel/vPlaceholder_51*
dtype0
Е
ReadVariableOp_51ReadVariableOptraining/Adam/fc1/kernel/v^AssignVariableOp_51*
dtype0*!
_output_shapes
:А─А
i
Placeholder_52Placeholder*
shape:         *
dtype0*#
_output_shapes
:         
^
AssignVariableOp_52AssignVariableOptraining/Adam/fc1/bias/vPlaceholder_52*
dtype0
}
ReadVariableOp_52ReadVariableOptraining/Adam/fc1/bias/v^AssignVariableOp_52*
dtype0*
_output_shapes	
:А
Г
Placeholder_53Placeholder*%
shape:                  *
dtype0*0
_output_shapes
:                  
`
AssignVariableOp_53AssignVariableOptraining/Adam/fc2/kernel/vPlaceholder_53*
dtype0
Д
ReadVariableOp_53ReadVariableOptraining/Adam/fc2/kernel/v^AssignVariableOp_53*
dtype0* 
_output_shapes
:
АА
i
Placeholder_54Placeholder*
shape:         *
dtype0*#
_output_shapes
:         
^
AssignVariableOp_54AssignVariableOptraining/Adam/fc2/bias/vPlaceholder_54*
dtype0
}
ReadVariableOp_54ReadVariableOptraining/Adam/fc2/bias/v^AssignVariableOp_54*
dtype0*
_output_shapes	
:А
Г
Placeholder_55Placeholder*%
shape:                  *
dtype0*0
_output_shapes
:                  
a
AssignVariableOp_55AssignVariableOptraining/Adam/fc2a/kernel/vPlaceholder_55*
dtype0
Е
ReadVariableOp_55ReadVariableOptraining/Adam/fc2a/kernel/v^AssignVariableOp_55*
dtype0* 
_output_shapes
:
АА
i
Placeholder_56Placeholder*
shape:         *
dtype0*#
_output_shapes
:         
_
AssignVariableOp_56AssignVariableOptraining/Adam/fc2a/bias/vPlaceholder_56*
dtype0
~
ReadVariableOp_56ReadVariableOptraining/Adam/fc2a/bias/v^AssignVariableOp_56*
dtype0*
_output_shapes	
:А
Г
Placeholder_57Placeholder*%
shape:                  *
dtype0*0
_output_shapes
:                  
`
AssignVariableOp_57AssignVariableOptraining/Adam/fc3/kernel/vPlaceholder_57*
dtype0
Д
ReadVariableOp_57ReadVariableOptraining/Adam/fc3/kernel/v^AssignVariableOp_57*
dtype0* 
_output_shapes
:
АА
i
Placeholder_58Placeholder*
shape:         *
dtype0*#
_output_shapes
:         
^
AssignVariableOp_58AssignVariableOptraining/Adam/fc3/bias/vPlaceholder_58*
dtype0
}
ReadVariableOp_58ReadVariableOptraining/Adam/fc3/bias/v^AssignVariableOp_58*
dtype0*
_output_shapes	
:А
Г
Placeholder_59Placeholder*%
shape:                  *
dtype0*0
_output_shapes
:                  
`
AssignVariableOp_59AssignVariableOptraining/Adam/fc4/kernel/vPlaceholder_59*
dtype0
Г
ReadVariableOp_59ReadVariableOptraining/Adam/fc4/kernel/v^AssignVariableOp_59*
dtype0*
_output_shapes
:	А@
i
Placeholder_60Placeholder*
shape:         *
dtype0*#
_output_shapes
:         
^
AssignVariableOp_60AssignVariableOptraining/Adam/fc4/bias/vPlaceholder_60*
dtype0
|
ReadVariableOp_60ReadVariableOptraining/Adam/fc4/bias/v^AssignVariableOp_60*
dtype0*
_output_shapes
:@
Г
Placeholder_61Placeholder*%
shape:                  *
dtype0*0
_output_shapes
:                  
d
AssignVariableOp_61AssignVariableOptraining/Adam/dense_1/kernel/vPlaceholder_61*
dtype0
Ж
ReadVariableOp_61ReadVariableOptraining/Adam/dense_1/kernel/v^AssignVariableOp_61*
dtype0*
_output_shapes

:@
i
Placeholder_62Placeholder*
shape:         *
dtype0*#
_output_shapes
:         
b
AssignVariableOp_62AssignVariableOptraining/Adam/dense_1/bias/vPlaceholder_62*
dtype0
А
ReadVariableOp_62ReadVariableOptraining/Adam/dense_1/bias/v^AssignVariableOp_62*
dtype0*
_output_shapes
:
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
dtype0*
_output_shapes
: 
e

save/ConstPlaceholderWithDefaultsave/filename*
shape: *
dtype0*
_output_shapes
: 
Д
save/StringJoin/inputs_1Const*<
value3B1 B+_temp_fedfd39e29914d04b01701b0d4ca4019/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
М
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
Х
save/SaveV2/tensor_namesConst"/device:CPU:0*╣
valueпBмCBblock1_conv1/biasBblock1_conv1/kernelBblock1_conv2/biasBblock1_conv2/kernelBblock2_conv1/biasBblock2_conv1/kernelBblock2_conv2/biasBblock2_conv2/kernelBblock3_conv1/biasBblock3_conv1/kernelBblock3_conv2/biasBblock3_conv2/kernelBblock3_conv3/biasBblock3_conv3/kernelBblock4_conv1/biasBblock4_conv1/kernelBblock4_conv2/biasBblock4_conv2/kernelBblock4_conv3/biasBblock4_conv3/kernelBblock5_conv1/biasBblock5_conv1/kernelBblock5_conv2/biasBblock5_conv2/kernelBblock5_conv3/biasBblock5_conv3/kernelBdense_1/biasBdense_1/kernelBfc1/biasB
fc1/kernelBfc2/biasB
fc2/kernelB	fc2a/biasBfc2a/kernelBfc3/biasB
fc3/kernelBfc4/biasB
fc4/kernelBtraining/Adam/beta_1Btraining/Adam/beta_2Btraining/Adam/decayBtraining/Adam/dense_1/bias/mBtraining/Adam/dense_1/bias/vBtraining/Adam/dense_1/kernel/mBtraining/Adam/dense_1/kernel/vBtraining/Adam/fc1/bias/mBtraining/Adam/fc1/bias/vBtraining/Adam/fc1/kernel/mBtraining/Adam/fc1/kernel/vBtraining/Adam/fc2/bias/mBtraining/Adam/fc2/bias/vBtraining/Adam/fc2/kernel/mBtraining/Adam/fc2/kernel/vBtraining/Adam/fc2a/bias/mBtraining/Adam/fc2a/bias/vBtraining/Adam/fc2a/kernel/mBtraining/Adam/fc2a/kernel/vBtraining/Adam/fc3/bias/mBtraining/Adam/fc3/bias/vBtraining/Adam/fc3/kernel/mBtraining/Adam/fc3/kernel/vBtraining/Adam/fc4/bias/mBtraining/Adam/fc4/bias/vBtraining/Adam/fc4/kernel/mBtraining/Adam/fc4/kernel/vBtraining/Adam/iterBtraining/Adam/learning_rate*
dtype0*
_output_shapes
:C
√
save/SaveV2/shape_and_slicesConst"/device:CPU:0*Ы
valueСBОCB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:C
е
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices%block1_conv1/bias/Read/ReadVariableOp'block1_conv1/kernel/Read/ReadVariableOp%block1_conv2/bias/Read/ReadVariableOp'block1_conv2/kernel/Read/ReadVariableOp%block2_conv1/bias/Read/ReadVariableOp'block2_conv1/kernel/Read/ReadVariableOp%block2_conv2/bias/Read/ReadVariableOp'block2_conv2/kernel/Read/ReadVariableOp%block3_conv1/bias/Read/ReadVariableOp'block3_conv1/kernel/Read/ReadVariableOp%block3_conv2/bias/Read/ReadVariableOp'block3_conv2/kernel/Read/ReadVariableOp%block3_conv3/bias/Read/ReadVariableOp'block3_conv3/kernel/Read/ReadVariableOp%block4_conv1/bias/Read/ReadVariableOp'block4_conv1/kernel/Read/ReadVariableOp%block4_conv2/bias/Read/ReadVariableOp'block4_conv2/kernel/Read/ReadVariableOp%block4_conv3/bias/Read/ReadVariableOp'block4_conv3/kernel/Read/ReadVariableOp%block5_conv1/bias/Read/ReadVariableOp'block5_conv1/kernel/Read/ReadVariableOp%block5_conv2/bias/Read/ReadVariableOp'block5_conv2/kernel/Read/ReadVariableOp%block5_conv3/bias/Read/ReadVariableOp'block5_conv3/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOpfc1/bias/Read/ReadVariableOpfc1/kernel/Read/ReadVariableOpfc2/bias/Read/ReadVariableOpfc2/kernel/Read/ReadVariableOpfc2a/bias/Read/ReadVariableOpfc2a/kernel/Read/ReadVariableOpfc3/bias/Read/ReadVariableOpfc3/kernel/Read/ReadVariableOpfc4/bias/Read/ReadVariableOpfc4/kernel/Read/ReadVariableOp(training/Adam/beta_1/Read/ReadVariableOp(training/Adam/beta_2/Read/ReadVariableOp'training/Adam/decay/Read/ReadVariableOp0training/Adam/dense_1/bias/m/Read/ReadVariableOp0training/Adam/dense_1/bias/v/Read/ReadVariableOp2training/Adam/dense_1/kernel/m/Read/ReadVariableOp2training/Adam/dense_1/kernel/v/Read/ReadVariableOp,training/Adam/fc1/bias/m/Read/ReadVariableOp,training/Adam/fc1/bias/v/Read/ReadVariableOp.training/Adam/fc1/kernel/m/Read/ReadVariableOp.training/Adam/fc1/kernel/v/Read/ReadVariableOp,training/Adam/fc2/bias/m/Read/ReadVariableOp,training/Adam/fc2/bias/v/Read/ReadVariableOp.training/Adam/fc2/kernel/m/Read/ReadVariableOp.training/Adam/fc2/kernel/v/Read/ReadVariableOp-training/Adam/fc2a/bias/m/Read/ReadVariableOp-training/Adam/fc2a/bias/v/Read/ReadVariableOp/training/Adam/fc2a/kernel/m/Read/ReadVariableOp/training/Adam/fc2a/kernel/v/Read/ReadVariableOp,training/Adam/fc3/bias/m/Read/ReadVariableOp,training/Adam/fc3/bias/v/Read/ReadVariableOp.training/Adam/fc3/kernel/m/Read/ReadVariableOp.training/Adam/fc3/kernel/v/Read/ReadVariableOp,training/Adam/fc4/bias/m/Read/ReadVariableOp,training/Adam/fc4/bias/v/Read/ReadVariableOp.training/Adam/fc4/kernel/m/Read/ReadVariableOp.training/Adam/fc4/kernel/v/Read/ReadVariableOp&training/Adam/iter/Read/ReadVariableOp/training/Adam/learning_rate/Read/ReadVariableOp"/device:CPU:0*Q
dtypesG
E2C	
а
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
м
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:
М
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(
Й
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
Ш
save/RestoreV2/tensor_namesConst"/device:CPU:0*╣
valueпBмCBblock1_conv1/biasBblock1_conv1/kernelBblock1_conv2/biasBblock1_conv2/kernelBblock2_conv1/biasBblock2_conv1/kernelBblock2_conv2/biasBblock2_conv2/kernelBblock3_conv1/biasBblock3_conv1/kernelBblock3_conv2/biasBblock3_conv2/kernelBblock3_conv3/biasBblock3_conv3/kernelBblock4_conv1/biasBblock4_conv1/kernelBblock4_conv2/biasBblock4_conv2/kernelBblock4_conv3/biasBblock4_conv3/kernelBblock5_conv1/biasBblock5_conv1/kernelBblock5_conv2/biasBblock5_conv2/kernelBblock5_conv3/biasBblock5_conv3/kernelBdense_1/biasBdense_1/kernelBfc1/biasB
fc1/kernelBfc2/biasB
fc2/kernelB	fc2a/biasBfc2a/kernelBfc3/biasB
fc3/kernelBfc4/biasB
fc4/kernelBtraining/Adam/beta_1Btraining/Adam/beta_2Btraining/Adam/decayBtraining/Adam/dense_1/bias/mBtraining/Adam/dense_1/bias/vBtraining/Adam/dense_1/kernel/mBtraining/Adam/dense_1/kernel/vBtraining/Adam/fc1/bias/mBtraining/Adam/fc1/bias/vBtraining/Adam/fc1/kernel/mBtraining/Adam/fc1/kernel/vBtraining/Adam/fc2/bias/mBtraining/Adam/fc2/bias/vBtraining/Adam/fc2/kernel/mBtraining/Adam/fc2/kernel/vBtraining/Adam/fc2a/bias/mBtraining/Adam/fc2a/bias/vBtraining/Adam/fc2a/kernel/mBtraining/Adam/fc2a/kernel/vBtraining/Adam/fc3/bias/mBtraining/Adam/fc3/bias/vBtraining/Adam/fc3/kernel/mBtraining/Adam/fc3/kernel/vBtraining/Adam/fc4/bias/mBtraining/Adam/fc4/bias/vBtraining/Adam/fc4/kernel/mBtraining/Adam/fc4/kernel/vBtraining/Adam/iterBtraining/Adam/learning_rate*
dtype0*
_output_shapes
:C
■
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*Ы
valueСBОCB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:C
ь
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*Q
dtypesG
E2C	*в
_output_shapesП
М:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
N
save/Identity_1Identitysave/RestoreV2*
T0*
_output_shapes
:
Z
save/AssignVariableOpAssignVariableOpblock1_conv1/biassave/Identity_1*
dtype0
P
save/Identity_2Identitysave/RestoreV2:1*
T0*
_output_shapes
:
^
save/AssignVariableOp_1AssignVariableOpblock1_conv1/kernelsave/Identity_2*
dtype0
P
save/Identity_3Identitysave/RestoreV2:2*
T0*
_output_shapes
:
\
save/AssignVariableOp_2AssignVariableOpblock1_conv2/biassave/Identity_3*
dtype0
P
save/Identity_4Identitysave/RestoreV2:3*
T0*
_output_shapes
:
^
save/AssignVariableOp_3AssignVariableOpblock1_conv2/kernelsave/Identity_4*
dtype0
P
save/Identity_5Identitysave/RestoreV2:4*
T0*
_output_shapes
:
\
save/AssignVariableOp_4AssignVariableOpblock2_conv1/biassave/Identity_5*
dtype0
P
save/Identity_6Identitysave/RestoreV2:5*
T0*
_output_shapes
:
^
save/AssignVariableOp_5AssignVariableOpblock2_conv1/kernelsave/Identity_6*
dtype0
P
save/Identity_7Identitysave/RestoreV2:6*
T0*
_output_shapes
:
\
save/AssignVariableOp_6AssignVariableOpblock2_conv2/biassave/Identity_7*
dtype0
P
save/Identity_8Identitysave/RestoreV2:7*
T0*
_output_shapes
:
^
save/AssignVariableOp_7AssignVariableOpblock2_conv2/kernelsave/Identity_8*
dtype0
P
save/Identity_9Identitysave/RestoreV2:8*
T0*
_output_shapes
:
\
save/AssignVariableOp_8AssignVariableOpblock3_conv1/biassave/Identity_9*
dtype0
Q
save/Identity_10Identitysave/RestoreV2:9*
T0*
_output_shapes
:
_
save/AssignVariableOp_9AssignVariableOpblock3_conv1/kernelsave/Identity_10*
dtype0
R
save/Identity_11Identitysave/RestoreV2:10*
T0*
_output_shapes
:
^
save/AssignVariableOp_10AssignVariableOpblock3_conv2/biassave/Identity_11*
dtype0
R
save/Identity_12Identitysave/RestoreV2:11*
T0*
_output_shapes
:
`
save/AssignVariableOp_11AssignVariableOpblock3_conv2/kernelsave/Identity_12*
dtype0
R
save/Identity_13Identitysave/RestoreV2:12*
T0*
_output_shapes
:
^
save/AssignVariableOp_12AssignVariableOpblock3_conv3/biassave/Identity_13*
dtype0
R
save/Identity_14Identitysave/RestoreV2:13*
T0*
_output_shapes
:
`
save/AssignVariableOp_13AssignVariableOpblock3_conv3/kernelsave/Identity_14*
dtype0
R
save/Identity_15Identitysave/RestoreV2:14*
T0*
_output_shapes
:
^
save/AssignVariableOp_14AssignVariableOpblock4_conv1/biassave/Identity_15*
dtype0
R
save/Identity_16Identitysave/RestoreV2:15*
T0*
_output_shapes
:
`
save/AssignVariableOp_15AssignVariableOpblock4_conv1/kernelsave/Identity_16*
dtype0
R
save/Identity_17Identitysave/RestoreV2:16*
T0*
_output_shapes
:
^
save/AssignVariableOp_16AssignVariableOpblock4_conv2/biassave/Identity_17*
dtype0
R
save/Identity_18Identitysave/RestoreV2:17*
T0*
_output_shapes
:
`
save/AssignVariableOp_17AssignVariableOpblock4_conv2/kernelsave/Identity_18*
dtype0
R
save/Identity_19Identitysave/RestoreV2:18*
T0*
_output_shapes
:
^
save/AssignVariableOp_18AssignVariableOpblock4_conv3/biassave/Identity_19*
dtype0
R
save/Identity_20Identitysave/RestoreV2:19*
T0*
_output_shapes
:
`
save/AssignVariableOp_19AssignVariableOpblock4_conv3/kernelsave/Identity_20*
dtype0
R
save/Identity_21Identitysave/RestoreV2:20*
T0*
_output_shapes
:
^
save/AssignVariableOp_20AssignVariableOpblock5_conv1/biassave/Identity_21*
dtype0
R
save/Identity_22Identitysave/RestoreV2:21*
T0*
_output_shapes
:
`
save/AssignVariableOp_21AssignVariableOpblock5_conv1/kernelsave/Identity_22*
dtype0
R
save/Identity_23Identitysave/RestoreV2:22*
T0*
_output_shapes
:
^
save/AssignVariableOp_22AssignVariableOpblock5_conv2/biassave/Identity_23*
dtype0
R
save/Identity_24Identitysave/RestoreV2:23*
T0*
_output_shapes
:
`
save/AssignVariableOp_23AssignVariableOpblock5_conv2/kernelsave/Identity_24*
dtype0
R
save/Identity_25Identitysave/RestoreV2:24*
T0*
_output_shapes
:
^
save/AssignVariableOp_24AssignVariableOpblock5_conv3/biassave/Identity_25*
dtype0
R
save/Identity_26Identitysave/RestoreV2:25*
T0*
_output_shapes
:
`
save/AssignVariableOp_25AssignVariableOpblock5_conv3/kernelsave/Identity_26*
dtype0
R
save/Identity_27Identitysave/RestoreV2:26*
T0*
_output_shapes
:
Y
save/AssignVariableOp_26AssignVariableOpdense_1/biassave/Identity_27*
dtype0
R
save/Identity_28Identitysave/RestoreV2:27*
T0*
_output_shapes
:
[
save/AssignVariableOp_27AssignVariableOpdense_1/kernelsave/Identity_28*
dtype0
R
save/Identity_29Identitysave/RestoreV2:28*
T0*
_output_shapes
:
U
save/AssignVariableOp_28AssignVariableOpfc1/biassave/Identity_29*
dtype0
R
save/Identity_30Identitysave/RestoreV2:29*
T0*
_output_shapes
:
W
save/AssignVariableOp_29AssignVariableOp
fc1/kernelsave/Identity_30*
dtype0
R
save/Identity_31Identitysave/RestoreV2:30*
T0*
_output_shapes
:
U
save/AssignVariableOp_30AssignVariableOpfc2/biassave/Identity_31*
dtype0
R
save/Identity_32Identitysave/RestoreV2:31*
T0*
_output_shapes
:
W
save/AssignVariableOp_31AssignVariableOp
fc2/kernelsave/Identity_32*
dtype0
R
save/Identity_33Identitysave/RestoreV2:32*
T0*
_output_shapes
:
V
save/AssignVariableOp_32AssignVariableOp	fc2a/biassave/Identity_33*
dtype0
R
save/Identity_34Identitysave/RestoreV2:33*
T0*
_output_shapes
:
X
save/AssignVariableOp_33AssignVariableOpfc2a/kernelsave/Identity_34*
dtype0
R
save/Identity_35Identitysave/RestoreV2:34*
T0*
_output_shapes
:
U
save/AssignVariableOp_34AssignVariableOpfc3/biassave/Identity_35*
dtype0
R
save/Identity_36Identitysave/RestoreV2:35*
T0*
_output_shapes
:
W
save/AssignVariableOp_35AssignVariableOp
fc3/kernelsave/Identity_36*
dtype0
R
save/Identity_37Identitysave/RestoreV2:36*
T0*
_output_shapes
:
U
save/AssignVariableOp_36AssignVariableOpfc4/biassave/Identity_37*
dtype0
R
save/Identity_38Identitysave/RestoreV2:37*
T0*
_output_shapes
:
W
save/AssignVariableOp_37AssignVariableOp
fc4/kernelsave/Identity_38*
dtype0
R
save/Identity_39Identitysave/RestoreV2:38*
T0*
_output_shapes
:
a
save/AssignVariableOp_38AssignVariableOptraining/Adam/beta_1save/Identity_39*
dtype0
R
save/Identity_40Identitysave/RestoreV2:39*
T0*
_output_shapes
:
a
save/AssignVariableOp_39AssignVariableOptraining/Adam/beta_2save/Identity_40*
dtype0
R
save/Identity_41Identitysave/RestoreV2:40*
T0*
_output_shapes
:
`
save/AssignVariableOp_40AssignVariableOptraining/Adam/decaysave/Identity_41*
dtype0
R
save/Identity_42Identitysave/RestoreV2:41*
T0*
_output_shapes
:
i
save/AssignVariableOp_41AssignVariableOptraining/Adam/dense_1/bias/msave/Identity_42*
dtype0
R
save/Identity_43Identitysave/RestoreV2:42*
T0*
_output_shapes
:
i
save/AssignVariableOp_42AssignVariableOptraining/Adam/dense_1/bias/vsave/Identity_43*
dtype0
R
save/Identity_44Identitysave/RestoreV2:43*
T0*
_output_shapes
:
k
save/AssignVariableOp_43AssignVariableOptraining/Adam/dense_1/kernel/msave/Identity_44*
dtype0
R
save/Identity_45Identitysave/RestoreV2:44*
T0*
_output_shapes
:
k
save/AssignVariableOp_44AssignVariableOptraining/Adam/dense_1/kernel/vsave/Identity_45*
dtype0
R
save/Identity_46Identitysave/RestoreV2:45*
T0*
_output_shapes
:
e
save/AssignVariableOp_45AssignVariableOptraining/Adam/fc1/bias/msave/Identity_46*
dtype0
R
save/Identity_47Identitysave/RestoreV2:46*
T0*
_output_shapes
:
e
save/AssignVariableOp_46AssignVariableOptraining/Adam/fc1/bias/vsave/Identity_47*
dtype0
R
save/Identity_48Identitysave/RestoreV2:47*
T0*
_output_shapes
:
g
save/AssignVariableOp_47AssignVariableOptraining/Adam/fc1/kernel/msave/Identity_48*
dtype0
R
save/Identity_49Identitysave/RestoreV2:48*
T0*
_output_shapes
:
g
save/AssignVariableOp_48AssignVariableOptraining/Adam/fc1/kernel/vsave/Identity_49*
dtype0
R
save/Identity_50Identitysave/RestoreV2:49*
T0*
_output_shapes
:
e
save/AssignVariableOp_49AssignVariableOptraining/Adam/fc2/bias/msave/Identity_50*
dtype0
R
save/Identity_51Identitysave/RestoreV2:50*
T0*
_output_shapes
:
e
save/AssignVariableOp_50AssignVariableOptraining/Adam/fc2/bias/vsave/Identity_51*
dtype0
R
save/Identity_52Identitysave/RestoreV2:51*
T0*
_output_shapes
:
g
save/AssignVariableOp_51AssignVariableOptraining/Adam/fc2/kernel/msave/Identity_52*
dtype0
R
save/Identity_53Identitysave/RestoreV2:52*
T0*
_output_shapes
:
g
save/AssignVariableOp_52AssignVariableOptraining/Adam/fc2/kernel/vsave/Identity_53*
dtype0
R
save/Identity_54Identitysave/RestoreV2:53*
T0*
_output_shapes
:
f
save/AssignVariableOp_53AssignVariableOptraining/Adam/fc2a/bias/msave/Identity_54*
dtype0
R
save/Identity_55Identitysave/RestoreV2:54*
T0*
_output_shapes
:
f
save/AssignVariableOp_54AssignVariableOptraining/Adam/fc2a/bias/vsave/Identity_55*
dtype0
R
save/Identity_56Identitysave/RestoreV2:55*
T0*
_output_shapes
:
h
save/AssignVariableOp_55AssignVariableOptraining/Adam/fc2a/kernel/msave/Identity_56*
dtype0
R
save/Identity_57Identitysave/RestoreV2:56*
T0*
_output_shapes
:
h
save/AssignVariableOp_56AssignVariableOptraining/Adam/fc2a/kernel/vsave/Identity_57*
dtype0
R
save/Identity_58Identitysave/RestoreV2:57*
T0*
_output_shapes
:
e
save/AssignVariableOp_57AssignVariableOptraining/Adam/fc3/bias/msave/Identity_58*
dtype0
R
save/Identity_59Identitysave/RestoreV2:58*
T0*
_output_shapes
:
e
save/AssignVariableOp_58AssignVariableOptraining/Adam/fc3/bias/vsave/Identity_59*
dtype0
R
save/Identity_60Identitysave/RestoreV2:59*
T0*
_output_shapes
:
g
save/AssignVariableOp_59AssignVariableOptraining/Adam/fc3/kernel/msave/Identity_60*
dtype0
R
save/Identity_61Identitysave/RestoreV2:60*
T0*
_output_shapes
:
g
save/AssignVariableOp_60AssignVariableOptraining/Adam/fc3/kernel/vsave/Identity_61*
dtype0
R
save/Identity_62Identitysave/RestoreV2:61*
T0*
_output_shapes
:
e
save/AssignVariableOp_61AssignVariableOptraining/Adam/fc4/bias/msave/Identity_62*
dtype0
R
save/Identity_63Identitysave/RestoreV2:62*
T0*
_output_shapes
:
e
save/AssignVariableOp_62AssignVariableOptraining/Adam/fc4/bias/vsave/Identity_63*
dtype0
R
save/Identity_64Identitysave/RestoreV2:63*
T0*
_output_shapes
:
g
save/AssignVariableOp_63AssignVariableOptraining/Adam/fc4/kernel/msave/Identity_64*
dtype0
R
save/Identity_65Identitysave/RestoreV2:64*
T0*
_output_shapes
:
g
save/AssignVariableOp_64AssignVariableOptraining/Adam/fc4/kernel/vsave/Identity_65*
dtype0
R
save/Identity_66Identitysave/RestoreV2:65*
T0	*
_output_shapes
:
_
save/AssignVariableOp_65AssignVariableOptraining/Adam/itersave/Identity_66*
dtype0	
R
save/Identity_67Identitysave/RestoreV2:66*
T0*
_output_shapes
:
h
save/AssignVariableOp_66AssignVariableOptraining/Adam/learning_ratesave/Identity_67*
dtype0
Я
save/restore_shardNoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_10^save/AssignVariableOp_11^save/AssignVariableOp_12^save/AssignVariableOp_13^save/AssignVariableOp_14^save/AssignVariableOp_15^save/AssignVariableOp_16^save/AssignVariableOp_17^save/AssignVariableOp_18^save/AssignVariableOp_19^save/AssignVariableOp_2^save/AssignVariableOp_20^save/AssignVariableOp_21^save/AssignVariableOp_22^save/AssignVariableOp_23^save/AssignVariableOp_24^save/AssignVariableOp_25^save/AssignVariableOp_26^save/AssignVariableOp_27^save/AssignVariableOp_28^save/AssignVariableOp_29^save/AssignVariableOp_3^save/AssignVariableOp_30^save/AssignVariableOp_31^save/AssignVariableOp_32^save/AssignVariableOp_33^save/AssignVariableOp_34^save/AssignVariableOp_35^save/AssignVariableOp_36^save/AssignVariableOp_37^save/AssignVariableOp_38^save/AssignVariableOp_39^save/AssignVariableOp_4^save/AssignVariableOp_40^save/AssignVariableOp_41^save/AssignVariableOp_42^save/AssignVariableOp_43^save/AssignVariableOp_44^save/AssignVariableOp_45^save/AssignVariableOp_46^save/AssignVariableOp_47^save/AssignVariableOp_48^save/AssignVariableOp_49^save/AssignVariableOp_5^save/AssignVariableOp_50^save/AssignVariableOp_51^save/AssignVariableOp_52^save/AssignVariableOp_53^save/AssignVariableOp_54^save/AssignVariableOp_55^save/AssignVariableOp_56^save/AssignVariableOp_57^save/AssignVariableOp_58^save/AssignVariableOp_59^save/AssignVariableOp_6^save/AssignVariableOp_60^save/AssignVariableOp_61^save/AssignVariableOp_62^save/AssignVariableOp_63^save/AssignVariableOp_64^save/AssignVariableOp_65^save/AssignVariableOp_66^save/AssignVariableOp_7^save/AssignVariableOp_8^save/AssignVariableOp_9
-
save/restore_allNoOp^save/restore_shard"Ж<
save/Const:0save/Identity:0save/restore_all (5 @F8"п'
trainable_variablesЧ'Ф'
Ф
block1_conv1/kernel:0block1_conv1/kernel/Assign)block1_conv1/kernel/Read/ReadVariableOp:0(20block1_conv1/kernel/Initializer/random_uniform:08
Г
block1_conv1/bias:0block1_conv1/bias/Assign'block1_conv1/bias/Read/ReadVariableOp:0(2%block1_conv1/bias/Initializer/zeros:08
Ф
block1_conv2/kernel:0block1_conv2/kernel/Assign)block1_conv2/kernel/Read/ReadVariableOp:0(20block1_conv2/kernel/Initializer/random_uniform:08
Г
block1_conv2/bias:0block1_conv2/bias/Assign'block1_conv2/bias/Read/ReadVariableOp:0(2%block1_conv2/bias/Initializer/zeros:08
Ф
block2_conv1/kernel:0block2_conv1/kernel/Assign)block2_conv1/kernel/Read/ReadVariableOp:0(20block2_conv1/kernel/Initializer/random_uniform:08
Г
block2_conv1/bias:0block2_conv1/bias/Assign'block2_conv1/bias/Read/ReadVariableOp:0(2%block2_conv1/bias/Initializer/zeros:08
Ф
block2_conv2/kernel:0block2_conv2/kernel/Assign)block2_conv2/kernel/Read/ReadVariableOp:0(20block2_conv2/kernel/Initializer/random_uniform:08
Г
block2_conv2/bias:0block2_conv2/bias/Assign'block2_conv2/bias/Read/ReadVariableOp:0(2%block2_conv2/bias/Initializer/zeros:08
Ф
block3_conv1/kernel:0block3_conv1/kernel/Assign)block3_conv1/kernel/Read/ReadVariableOp:0(20block3_conv1/kernel/Initializer/random_uniform:08
Г
block3_conv1/bias:0block3_conv1/bias/Assign'block3_conv1/bias/Read/ReadVariableOp:0(2%block3_conv1/bias/Initializer/zeros:08
Ф
block3_conv2/kernel:0block3_conv2/kernel/Assign)block3_conv2/kernel/Read/ReadVariableOp:0(20block3_conv2/kernel/Initializer/random_uniform:08
Г
block3_conv2/bias:0block3_conv2/bias/Assign'block3_conv2/bias/Read/ReadVariableOp:0(2%block3_conv2/bias/Initializer/zeros:08
Ф
block3_conv3/kernel:0block3_conv3/kernel/Assign)block3_conv3/kernel/Read/ReadVariableOp:0(20block3_conv3/kernel/Initializer/random_uniform:08
Г
block3_conv3/bias:0block3_conv3/bias/Assign'block3_conv3/bias/Read/ReadVariableOp:0(2%block3_conv3/bias/Initializer/zeros:08
Ф
block4_conv1/kernel:0block4_conv1/kernel/Assign)block4_conv1/kernel/Read/ReadVariableOp:0(20block4_conv1/kernel/Initializer/random_uniform:08
Г
block4_conv1/bias:0block4_conv1/bias/Assign'block4_conv1/bias/Read/ReadVariableOp:0(2%block4_conv1/bias/Initializer/zeros:08
Ф
block4_conv2/kernel:0block4_conv2/kernel/Assign)block4_conv2/kernel/Read/ReadVariableOp:0(20block4_conv2/kernel/Initializer/random_uniform:08
Г
block4_conv2/bias:0block4_conv2/bias/Assign'block4_conv2/bias/Read/ReadVariableOp:0(2%block4_conv2/bias/Initializer/zeros:08
Ф
block4_conv3/kernel:0block4_conv3/kernel/Assign)block4_conv3/kernel/Read/ReadVariableOp:0(20block4_conv3/kernel/Initializer/random_uniform:08
Г
block4_conv3/bias:0block4_conv3/bias/Assign'block4_conv3/bias/Read/ReadVariableOp:0(2%block4_conv3/bias/Initializer/zeros:08
Ф
block5_conv1/kernel:0block5_conv1/kernel/Assign)block5_conv1/kernel/Read/ReadVariableOp:0(20block5_conv1/kernel/Initializer/random_uniform:08
Г
block5_conv1/bias:0block5_conv1/bias/Assign'block5_conv1/bias/Read/ReadVariableOp:0(2%block5_conv1/bias/Initializer/zeros:08
Ф
block5_conv2/kernel:0block5_conv2/kernel/Assign)block5_conv2/kernel/Read/ReadVariableOp:0(20block5_conv2/kernel/Initializer/random_uniform:08
Г
block5_conv2/bias:0block5_conv2/bias/Assign'block5_conv2/bias/Read/ReadVariableOp:0(2%block5_conv2/bias/Initializer/zeros:08
Ф
block5_conv3/kernel:0block5_conv3/kernel/Assign)block5_conv3/kernel/Read/ReadVariableOp:0(20block5_conv3/kernel/Initializer/random_uniform:08
Г
block5_conv3/bias:0block5_conv3/bias/Assign'block5_conv3/bias/Read/ReadVariableOp:0(2%block5_conv3/bias/Initializer/zeros:08
p
fc1/kernel:0fc1/kernel/Assign fc1/kernel/Read/ReadVariableOp:0(2'fc1/kernel/Initializer/random_uniform:08
_

fc1/bias:0fc1/bias/Assignfc1/bias/Read/ReadVariableOp:0(2fc1/bias/Initializer/zeros:08
p
fc2/kernel:0fc2/kernel/Assign fc2/kernel/Read/ReadVariableOp:0(2'fc2/kernel/Initializer/random_uniform:08
_

fc2/bias:0fc2/bias/Assignfc2/bias/Read/ReadVariableOp:0(2fc2/bias/Initializer/zeros:08
t
fc2a/kernel:0fc2a/kernel/Assign!fc2a/kernel/Read/ReadVariableOp:0(2(fc2a/kernel/Initializer/random_uniform:08
c
fc2a/bias:0fc2a/bias/Assignfc2a/bias/Read/ReadVariableOp:0(2fc2a/bias/Initializer/zeros:08
p
fc3/kernel:0fc3/kernel/Assign fc3/kernel/Read/ReadVariableOp:0(2'fc3/kernel/Initializer/random_uniform:08
_

fc3/bias:0fc3/bias/Assignfc3/bias/Read/ReadVariableOp:0(2fc3/bias/Initializer/zeros:08
p
fc4/kernel:0fc4/kernel/Assign fc4/kernel/Read/ReadVariableOp:0(2'fc4/kernel/Initializer/random_uniform:08
_

fc4/bias:0fc4/bias/Assignfc4/bias/Read/ReadVariableOp:0(2fc4/bias/Initializer/zeros:08
А
dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(2+dense_1/kernel/Initializer/random_uniform:08
o
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2 dense_1/bias/Initializer/zeros:08"▀L
	variables╤L╬L
Ф
block1_conv1/kernel:0block1_conv1/kernel/Assign)block1_conv1/kernel/Read/ReadVariableOp:0(20block1_conv1/kernel/Initializer/random_uniform:08
Г
block1_conv1/bias:0block1_conv1/bias/Assign'block1_conv1/bias/Read/ReadVariableOp:0(2%block1_conv1/bias/Initializer/zeros:08
Ф
block1_conv2/kernel:0block1_conv2/kernel/Assign)block1_conv2/kernel/Read/ReadVariableOp:0(20block1_conv2/kernel/Initializer/random_uniform:08
Г
block1_conv2/bias:0block1_conv2/bias/Assign'block1_conv2/bias/Read/ReadVariableOp:0(2%block1_conv2/bias/Initializer/zeros:08
Ф
block2_conv1/kernel:0block2_conv1/kernel/Assign)block2_conv1/kernel/Read/ReadVariableOp:0(20block2_conv1/kernel/Initializer/random_uniform:08
Г
block2_conv1/bias:0block2_conv1/bias/Assign'block2_conv1/bias/Read/ReadVariableOp:0(2%block2_conv1/bias/Initializer/zeros:08
Ф
block2_conv2/kernel:0block2_conv2/kernel/Assign)block2_conv2/kernel/Read/ReadVariableOp:0(20block2_conv2/kernel/Initializer/random_uniform:08
Г
block2_conv2/bias:0block2_conv2/bias/Assign'block2_conv2/bias/Read/ReadVariableOp:0(2%block2_conv2/bias/Initializer/zeros:08
Ф
block3_conv1/kernel:0block3_conv1/kernel/Assign)block3_conv1/kernel/Read/ReadVariableOp:0(20block3_conv1/kernel/Initializer/random_uniform:08
Г
block3_conv1/bias:0block3_conv1/bias/Assign'block3_conv1/bias/Read/ReadVariableOp:0(2%block3_conv1/bias/Initializer/zeros:08
Ф
block3_conv2/kernel:0block3_conv2/kernel/Assign)block3_conv2/kernel/Read/ReadVariableOp:0(20block3_conv2/kernel/Initializer/random_uniform:08
Г
block3_conv2/bias:0block3_conv2/bias/Assign'block3_conv2/bias/Read/ReadVariableOp:0(2%block3_conv2/bias/Initializer/zeros:08
Ф
block3_conv3/kernel:0block3_conv3/kernel/Assign)block3_conv3/kernel/Read/ReadVariableOp:0(20block3_conv3/kernel/Initializer/random_uniform:08
Г
block3_conv3/bias:0block3_conv3/bias/Assign'block3_conv3/bias/Read/ReadVariableOp:0(2%block3_conv3/bias/Initializer/zeros:08
Ф
block4_conv1/kernel:0block4_conv1/kernel/Assign)block4_conv1/kernel/Read/ReadVariableOp:0(20block4_conv1/kernel/Initializer/random_uniform:08
Г
block4_conv1/bias:0block4_conv1/bias/Assign'block4_conv1/bias/Read/ReadVariableOp:0(2%block4_conv1/bias/Initializer/zeros:08
Ф
block4_conv2/kernel:0block4_conv2/kernel/Assign)block4_conv2/kernel/Read/ReadVariableOp:0(20block4_conv2/kernel/Initializer/random_uniform:08
Г
block4_conv2/bias:0block4_conv2/bias/Assign'block4_conv2/bias/Read/ReadVariableOp:0(2%block4_conv2/bias/Initializer/zeros:08
Ф
block4_conv3/kernel:0block4_conv3/kernel/Assign)block4_conv3/kernel/Read/ReadVariableOp:0(20block4_conv3/kernel/Initializer/random_uniform:08
Г
block4_conv3/bias:0block4_conv3/bias/Assign'block4_conv3/bias/Read/ReadVariableOp:0(2%block4_conv3/bias/Initializer/zeros:08
Ф
block5_conv1/kernel:0block5_conv1/kernel/Assign)block5_conv1/kernel/Read/ReadVariableOp:0(20block5_conv1/kernel/Initializer/random_uniform:08
Г
block5_conv1/bias:0block5_conv1/bias/Assign'block5_conv1/bias/Read/ReadVariableOp:0(2%block5_conv1/bias/Initializer/zeros:08
Ф
block5_conv2/kernel:0block5_conv2/kernel/Assign)block5_conv2/kernel/Read/ReadVariableOp:0(20block5_conv2/kernel/Initializer/random_uniform:08
Г
block5_conv2/bias:0block5_conv2/bias/Assign'block5_conv2/bias/Read/ReadVariableOp:0(2%block5_conv2/bias/Initializer/zeros:08
Ф
block5_conv3/kernel:0block5_conv3/kernel/Assign)block5_conv3/kernel/Read/ReadVariableOp:0(20block5_conv3/kernel/Initializer/random_uniform:08
Г
block5_conv3/bias:0block5_conv3/bias/Assign'block5_conv3/bias/Read/ReadVariableOp:0(2%block5_conv3/bias/Initializer/zeros:08
p
fc1/kernel:0fc1/kernel/Assign fc1/kernel/Read/ReadVariableOp:0(2'fc1/kernel/Initializer/random_uniform:08
_

fc1/bias:0fc1/bias/Assignfc1/bias/Read/ReadVariableOp:0(2fc1/bias/Initializer/zeros:08
p
fc2/kernel:0fc2/kernel/Assign fc2/kernel/Read/ReadVariableOp:0(2'fc2/kernel/Initializer/random_uniform:08
_

fc2/bias:0fc2/bias/Assignfc2/bias/Read/ReadVariableOp:0(2fc2/bias/Initializer/zeros:08
t
fc2a/kernel:0fc2a/kernel/Assign!fc2a/kernel/Read/ReadVariableOp:0(2(fc2a/kernel/Initializer/random_uniform:08
c
fc2a/bias:0fc2a/bias/Assignfc2a/bias/Read/ReadVariableOp:0(2fc2a/bias/Initializer/zeros:08
p
fc3/kernel:0fc3/kernel/Assign fc3/kernel/Read/ReadVariableOp:0(2'fc3/kernel/Initializer/random_uniform:08
_

fc3/bias:0fc3/bias/Assignfc3/bias/Read/ReadVariableOp:0(2fc3/bias/Initializer/zeros:08
p
fc4/kernel:0fc4/kernel/Assign fc4/kernel/Read/ReadVariableOp:0(2'fc4/kernel/Initializer/random_uniform:08
_

fc4/bias:0fc4/bias/Assignfc4/bias/Read/ReadVariableOp:0(2fc4/bias/Initializer/zeros:08
А
dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(2+dense_1/kernel/Initializer/random_uniform:08
o
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2 dense_1/bias/Initializer/zeros:08
З
training/Adam/iter:0training/Adam/iter/Assign(training/Adam/iter/Read/ReadVariableOp:0(2&training/Adam/iter/Initializer/zeros:0H
Ч
training/Adam/beta_1:0training/Adam/beta_1/Assign*training/Adam/beta_1/Read/ReadVariableOp:0(20training/Adam/beta_1/Initializer/initial_value:0H
Ч
training/Adam/beta_2:0training/Adam/beta_2/Assign*training/Adam/beta_2/Read/ReadVariableOp:0(20training/Adam/beta_2/Initializer/initial_value:0H
У
training/Adam/decay:0training/Adam/decay/Assign)training/Adam/decay/Read/ReadVariableOp:0(2/training/Adam/decay/Initializer/initial_value:0H
│
training/Adam/learning_rate:0"training/Adam/learning_rate/Assign1training/Adam/learning_rate/Read/ReadVariableOp:0(27training/Adam/learning_rate/Initializer/initial_value:0H
е
training/Adam/fc1/kernel/m:0!training/Adam/fc1/kernel/m/Assign0training/Adam/fc1/kernel/m/Read/ReadVariableOp:0(2.training/Adam/fc1/kernel/m/Initializer/zeros:0
Э
training/Adam/fc1/bias/m:0training/Adam/fc1/bias/m/Assign.training/Adam/fc1/bias/m/Read/ReadVariableOp:0(2,training/Adam/fc1/bias/m/Initializer/zeros:0
е
training/Adam/fc2/kernel/m:0!training/Adam/fc2/kernel/m/Assign0training/Adam/fc2/kernel/m/Read/ReadVariableOp:0(2.training/Adam/fc2/kernel/m/Initializer/zeros:0
Э
training/Adam/fc2/bias/m:0training/Adam/fc2/bias/m/Assign.training/Adam/fc2/bias/m/Read/ReadVariableOp:0(2,training/Adam/fc2/bias/m/Initializer/zeros:0
й
training/Adam/fc2a/kernel/m:0"training/Adam/fc2a/kernel/m/Assign1training/Adam/fc2a/kernel/m/Read/ReadVariableOp:0(2/training/Adam/fc2a/kernel/m/Initializer/zeros:0
б
training/Adam/fc2a/bias/m:0 training/Adam/fc2a/bias/m/Assign/training/Adam/fc2a/bias/m/Read/ReadVariableOp:0(2-training/Adam/fc2a/bias/m/Initializer/zeros:0
е
training/Adam/fc3/kernel/m:0!training/Adam/fc3/kernel/m/Assign0training/Adam/fc3/kernel/m/Read/ReadVariableOp:0(2.training/Adam/fc3/kernel/m/Initializer/zeros:0
Э
training/Adam/fc3/bias/m:0training/Adam/fc3/bias/m/Assign.training/Adam/fc3/bias/m/Read/ReadVariableOp:0(2,training/Adam/fc3/bias/m/Initializer/zeros:0
е
training/Adam/fc4/kernel/m:0!training/Adam/fc4/kernel/m/Assign0training/Adam/fc4/kernel/m/Read/ReadVariableOp:0(2.training/Adam/fc4/kernel/m/Initializer/zeros:0
Э
training/Adam/fc4/bias/m:0training/Adam/fc4/bias/m/Assign.training/Adam/fc4/bias/m/Read/ReadVariableOp:0(2,training/Adam/fc4/bias/m/Initializer/zeros:0
╡
 training/Adam/dense_1/kernel/m:0%training/Adam/dense_1/kernel/m/Assign4training/Adam/dense_1/kernel/m/Read/ReadVariableOp:0(22training/Adam/dense_1/kernel/m/Initializer/zeros:0
н
training/Adam/dense_1/bias/m:0#training/Adam/dense_1/bias/m/Assign2training/Adam/dense_1/bias/m/Read/ReadVariableOp:0(20training/Adam/dense_1/bias/m/Initializer/zeros:0
е
training/Adam/fc1/kernel/v:0!training/Adam/fc1/kernel/v/Assign0training/Adam/fc1/kernel/v/Read/ReadVariableOp:0(2.training/Adam/fc1/kernel/v/Initializer/zeros:0
Э
training/Adam/fc1/bias/v:0training/Adam/fc1/bias/v/Assign.training/Adam/fc1/bias/v/Read/ReadVariableOp:0(2,training/Adam/fc1/bias/v/Initializer/zeros:0
е
training/Adam/fc2/kernel/v:0!training/Adam/fc2/kernel/v/Assign0training/Adam/fc2/kernel/v/Read/ReadVariableOp:0(2.training/Adam/fc2/kernel/v/Initializer/zeros:0
Э
training/Adam/fc2/bias/v:0training/Adam/fc2/bias/v/Assign.training/Adam/fc2/bias/v/Read/ReadVariableOp:0(2,training/Adam/fc2/bias/v/Initializer/zeros:0
й
training/Adam/fc2a/kernel/v:0"training/Adam/fc2a/kernel/v/Assign1training/Adam/fc2a/kernel/v/Read/ReadVariableOp:0(2/training/Adam/fc2a/kernel/v/Initializer/zeros:0
б
training/Adam/fc2a/bias/v:0 training/Adam/fc2a/bias/v/Assign/training/Adam/fc2a/bias/v/Read/ReadVariableOp:0(2-training/Adam/fc2a/bias/v/Initializer/zeros:0
е
training/Adam/fc3/kernel/v:0!training/Adam/fc3/kernel/v/Assign0training/Adam/fc3/kernel/v/Read/ReadVariableOp:0(2.training/Adam/fc3/kernel/v/Initializer/zeros:0
Э
training/Adam/fc3/bias/v:0training/Adam/fc3/bias/v/Assign.training/Adam/fc3/bias/v/Read/ReadVariableOp:0(2,training/Adam/fc3/bias/v/Initializer/zeros:0
е
training/Adam/fc4/kernel/v:0!training/Adam/fc4/kernel/v/Assign0training/Adam/fc4/kernel/v/Read/ReadVariableOp:0(2.training/Adam/fc4/kernel/v/Initializer/zeros:0
Э
training/Adam/fc4/bias/v:0training/Adam/fc4/bias/v/Assign.training/Adam/fc4/bias/v/Read/ReadVariableOp:0(2,training/Adam/fc4/bias/v/Initializer/zeros:0
╡
 training/Adam/dense_1/kernel/v:0%training/Adam/dense_1/kernel/v/Assign4training/Adam/dense_1/kernel/v/Read/ReadVariableOp:0(22training/Adam/dense_1/kernel/v/Initializer/zeros:0
н
training/Adam/dense_1/bias/v:0#training/Adam/dense_1/bias/v/Assign2training/Adam/dense_1/bias/v/Read/ReadVariableOp:0(20training/Adam/dense_1/bias/v/Initializer/zeros:0*Ъ
serving_defaultЖ
4
images*
	input_1:0         рр2
scores(
dense_1/Softmax:0         tensorflow/serving/predict