??/
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
l
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2		"
adj_xbool( "
adj_ybool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
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
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
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
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
d
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:

2	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
-
Sqrt
x"T
y"T"
Ttype:

2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
?
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
<
Sub
x"T
y"T
z"T"
Ttype:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-0-ga4dfb8d1a718??+
|
dense_102/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*!
shared_namedense_102/kernel
u
$dense_102/kernel/Read/ReadVariableOpReadVariableOpdense_102/kernel*
_output_shapes

:@*
dtype0
t
dense_102/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_102/bias
m
"dense_102/bias/Read/ReadVariableOpReadVariableOpdense_102/bias*
_output_shapes
:*
dtype0
|
dense_103/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_103/kernel
u
$dense_103/kernel/Read/ReadVariableOpReadVariableOpdense_103/kernel*
_output_shapes

:*
dtype0
t
dense_103/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_103/bias
m
"dense_103/bias/Read/ReadVariableOpReadVariableOpdense_103/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
6token_and_position_embedding_9/embedding_20/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@@*G
shared_name86token_and_position_embedding_9/embedding_20/embeddings
?
Jtoken_and_position_embedding_9/embedding_20/embeddings/Read/ReadVariableOpReadVariableOp6token_and_position_embedding_9/embedding_20/embeddings*
_output_shapes
:	?@@*
dtype0
?
6token_and_position_embedding_9/embedding_21/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d@*G
shared_name86token_and_position_embedding_9/embedding_21/embeddings
?
Jtoken_and_position_embedding_9/embedding_21/embeddings/Read/ReadVariableOpReadVariableOp6token_and_position_embedding_9/embedding_21/embeddings*
_output_shapes

:d@*
dtype0
?
:transformer_block_8/multi_head_attention_8/dense_96/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*K
shared_name<:transformer_block_8/multi_head_attention_8/dense_96/kernel
?
Ntransformer_block_8/multi_head_attention_8/dense_96/kernel/Read/ReadVariableOpReadVariableOp:transformer_block_8/multi_head_attention_8/dense_96/kernel*
_output_shapes

:@@*
dtype0
?
8transformer_block_8/multi_head_attention_8/dense_96/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*I
shared_name:8transformer_block_8/multi_head_attention_8/dense_96/bias
?
Ltransformer_block_8/multi_head_attention_8/dense_96/bias/Read/ReadVariableOpReadVariableOp8transformer_block_8/multi_head_attention_8/dense_96/bias*
_output_shapes
:@*
dtype0
?
:transformer_block_8/multi_head_attention_8/dense_97/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*K
shared_name<:transformer_block_8/multi_head_attention_8/dense_97/kernel
?
Ntransformer_block_8/multi_head_attention_8/dense_97/kernel/Read/ReadVariableOpReadVariableOp:transformer_block_8/multi_head_attention_8/dense_97/kernel*
_output_shapes

:@@*
dtype0
?
8transformer_block_8/multi_head_attention_8/dense_97/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*I
shared_name:8transformer_block_8/multi_head_attention_8/dense_97/bias
?
Ltransformer_block_8/multi_head_attention_8/dense_97/bias/Read/ReadVariableOpReadVariableOp8transformer_block_8/multi_head_attention_8/dense_97/bias*
_output_shapes
:@*
dtype0
?
:transformer_block_8/multi_head_attention_8/dense_98/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*K
shared_name<:transformer_block_8/multi_head_attention_8/dense_98/kernel
?
Ntransformer_block_8/multi_head_attention_8/dense_98/kernel/Read/ReadVariableOpReadVariableOp:transformer_block_8/multi_head_attention_8/dense_98/kernel*
_output_shapes

:@@*
dtype0
?
8transformer_block_8/multi_head_attention_8/dense_98/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*I
shared_name:8transformer_block_8/multi_head_attention_8/dense_98/bias
?
Ltransformer_block_8/multi_head_attention_8/dense_98/bias/Read/ReadVariableOpReadVariableOp8transformer_block_8/multi_head_attention_8/dense_98/bias*
_output_shapes
:@*
dtype0
?
:transformer_block_8/multi_head_attention_8/dense_99/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*K
shared_name<:transformer_block_8/multi_head_attention_8/dense_99/kernel
?
Ntransformer_block_8/multi_head_attention_8/dense_99/kernel/Read/ReadVariableOpReadVariableOp:transformer_block_8/multi_head_attention_8/dense_99/kernel*
_output_shapes

:@@*
dtype0
?
8transformer_block_8/multi_head_attention_8/dense_99/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*I
shared_name:8transformer_block_8/multi_head_attention_8/dense_99/bias
?
Ltransformer_block_8/multi_head_attention_8/dense_99/bias/Read/ReadVariableOpReadVariableOp8transformer_block_8/multi_head_attention_8/dense_99/bias*
_output_shapes
:@*
dtype0
|
dense_100/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *!
shared_namedense_100/kernel
u
$dense_100/kernel/Read/ReadVariableOpReadVariableOpdense_100/kernel*
_output_shapes

:@ *
dtype0
t
dense_100/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_100/bias
m
"dense_100/bias/Read/ReadVariableOpReadVariableOpdense_100/bias*
_output_shapes
: *
dtype0
|
dense_101/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*!
shared_namedense_101/kernel
u
$dense_101/kernel/Read/ReadVariableOpReadVariableOpdense_101/kernel*
_output_shapes

: @*
dtype0
t
dense_101/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_101/bias
m
"dense_101/bias/Read/ReadVariableOpReadVariableOpdense_101/bias*
_output_shapes
:@*
dtype0
?
0transformer_block_8/layer_normalization_26/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*A
shared_name20transformer_block_8/layer_normalization_26/gamma
?
Dtransformer_block_8/layer_normalization_26/gamma/Read/ReadVariableOpReadVariableOp0transformer_block_8/layer_normalization_26/gamma*
_output_shapes
:@*
dtype0
?
/transformer_block_8/layer_normalization_26/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*@
shared_name1/transformer_block_8/layer_normalization_26/beta
?
Ctransformer_block_8/layer_normalization_26/beta/Read/ReadVariableOpReadVariableOp/transformer_block_8/layer_normalization_26/beta*
_output_shapes
:@*
dtype0
?
0transformer_block_8/layer_normalization_27/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*A
shared_name20transformer_block_8/layer_normalization_27/gamma
?
Dtransformer_block_8/layer_normalization_27/gamma/Read/ReadVariableOpReadVariableOp0transformer_block_8/layer_normalization_27/gamma*
_output_shapes
:@*
dtype0
?
/transformer_block_8/layer_normalization_27/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*@
shared_name1/transformer_block_8/layer_normalization_27/beta
?
Ctransformer_block_8/layer_normalization_27/beta/Read/ReadVariableOpReadVariableOp/transformer_block_8/layer_normalization_27/beta*
_output_shapes
:@*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/dense_102/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*(
shared_nameAdam/dense_102/kernel/m
?
+Adam/dense_102/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_102/kernel/m*
_output_shapes

:@*
dtype0
?
Adam/dense_102/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_102/bias/m
{
)Adam/dense_102/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_102/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_103/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_103/kernel/m
?
+Adam/dense_103/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_103/kernel/m*
_output_shapes

:*
dtype0
?
Adam/dense_103/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_103/bias/m
{
)Adam/dense_103/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_103/bias/m*
_output_shapes
:*
dtype0
?
=Adam/token_and_position_embedding_9/embedding_20/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@@*N
shared_name?=Adam/token_and_position_embedding_9/embedding_20/embeddings/m
?
QAdam/token_and_position_embedding_9/embedding_20/embeddings/m/Read/ReadVariableOpReadVariableOp=Adam/token_and_position_embedding_9/embedding_20/embeddings/m*
_output_shapes
:	?@@*
dtype0
?
=Adam/token_and_position_embedding_9/embedding_21/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d@*N
shared_name?=Adam/token_and_position_embedding_9/embedding_21/embeddings/m
?
QAdam/token_and_position_embedding_9/embedding_21/embeddings/m/Read/ReadVariableOpReadVariableOp=Adam/token_and_position_embedding_9/embedding_21/embeddings/m*
_output_shapes

:d@*
dtype0
?
AAdam/transformer_block_8/multi_head_attention_8/dense_96/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*R
shared_nameCAAdam/transformer_block_8/multi_head_attention_8/dense_96/kernel/m
?
UAdam/transformer_block_8/multi_head_attention_8/dense_96/kernel/m/Read/ReadVariableOpReadVariableOpAAdam/transformer_block_8/multi_head_attention_8/dense_96/kernel/m*
_output_shapes

:@@*
dtype0
?
?Adam/transformer_block_8/multi_head_attention_8/dense_96/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*P
shared_nameA?Adam/transformer_block_8/multi_head_attention_8/dense_96/bias/m
?
SAdam/transformer_block_8/multi_head_attention_8/dense_96/bias/m/Read/ReadVariableOpReadVariableOp?Adam/transformer_block_8/multi_head_attention_8/dense_96/bias/m*
_output_shapes
:@*
dtype0
?
AAdam/transformer_block_8/multi_head_attention_8/dense_97/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*R
shared_nameCAAdam/transformer_block_8/multi_head_attention_8/dense_97/kernel/m
?
UAdam/transformer_block_8/multi_head_attention_8/dense_97/kernel/m/Read/ReadVariableOpReadVariableOpAAdam/transformer_block_8/multi_head_attention_8/dense_97/kernel/m*
_output_shapes

:@@*
dtype0
?
?Adam/transformer_block_8/multi_head_attention_8/dense_97/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*P
shared_nameA?Adam/transformer_block_8/multi_head_attention_8/dense_97/bias/m
?
SAdam/transformer_block_8/multi_head_attention_8/dense_97/bias/m/Read/ReadVariableOpReadVariableOp?Adam/transformer_block_8/multi_head_attention_8/dense_97/bias/m*
_output_shapes
:@*
dtype0
?
AAdam/transformer_block_8/multi_head_attention_8/dense_98/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*R
shared_nameCAAdam/transformer_block_8/multi_head_attention_8/dense_98/kernel/m
?
UAdam/transformer_block_8/multi_head_attention_8/dense_98/kernel/m/Read/ReadVariableOpReadVariableOpAAdam/transformer_block_8/multi_head_attention_8/dense_98/kernel/m*
_output_shapes

:@@*
dtype0
?
?Adam/transformer_block_8/multi_head_attention_8/dense_98/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*P
shared_nameA?Adam/transformer_block_8/multi_head_attention_8/dense_98/bias/m
?
SAdam/transformer_block_8/multi_head_attention_8/dense_98/bias/m/Read/ReadVariableOpReadVariableOp?Adam/transformer_block_8/multi_head_attention_8/dense_98/bias/m*
_output_shapes
:@*
dtype0
?
AAdam/transformer_block_8/multi_head_attention_8/dense_99/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*R
shared_nameCAAdam/transformer_block_8/multi_head_attention_8/dense_99/kernel/m
?
UAdam/transformer_block_8/multi_head_attention_8/dense_99/kernel/m/Read/ReadVariableOpReadVariableOpAAdam/transformer_block_8/multi_head_attention_8/dense_99/kernel/m*
_output_shapes

:@@*
dtype0
?
?Adam/transformer_block_8/multi_head_attention_8/dense_99/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*P
shared_nameA?Adam/transformer_block_8/multi_head_attention_8/dense_99/bias/m
?
SAdam/transformer_block_8/multi_head_attention_8/dense_99/bias/m/Read/ReadVariableOpReadVariableOp?Adam/transformer_block_8/multi_head_attention_8/dense_99/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_100/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *(
shared_nameAdam/dense_100/kernel/m
?
+Adam/dense_100/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_100/kernel/m*
_output_shapes

:@ *
dtype0
?
Adam/dense_100/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/dense_100/bias/m
{
)Adam/dense_100/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_100/bias/m*
_output_shapes
: *
dtype0
?
Adam/dense_101/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*(
shared_nameAdam/dense_101/kernel/m
?
+Adam/dense_101/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_101/kernel/m*
_output_shapes

: @*
dtype0
?
Adam/dense_101/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/dense_101/bias/m
{
)Adam/dense_101/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_101/bias/m*
_output_shapes
:@*
dtype0
?
7Adam/transformer_block_8/layer_normalization_26/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*H
shared_name97Adam/transformer_block_8/layer_normalization_26/gamma/m
?
KAdam/transformer_block_8/layer_normalization_26/gamma/m/Read/ReadVariableOpReadVariableOp7Adam/transformer_block_8/layer_normalization_26/gamma/m*
_output_shapes
:@*
dtype0
?
6Adam/transformer_block_8/layer_normalization_26/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*G
shared_name86Adam/transformer_block_8/layer_normalization_26/beta/m
?
JAdam/transformer_block_8/layer_normalization_26/beta/m/Read/ReadVariableOpReadVariableOp6Adam/transformer_block_8/layer_normalization_26/beta/m*
_output_shapes
:@*
dtype0
?
7Adam/transformer_block_8/layer_normalization_27/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*H
shared_name97Adam/transformer_block_8/layer_normalization_27/gamma/m
?
KAdam/transformer_block_8/layer_normalization_27/gamma/m/Read/ReadVariableOpReadVariableOp7Adam/transformer_block_8/layer_normalization_27/gamma/m*
_output_shapes
:@*
dtype0
?
6Adam/transformer_block_8/layer_normalization_27/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*G
shared_name86Adam/transformer_block_8/layer_normalization_27/beta/m
?
JAdam/transformer_block_8/layer_normalization_27/beta/m/Read/ReadVariableOpReadVariableOp6Adam/transformer_block_8/layer_normalization_27/beta/m*
_output_shapes
:@*
dtype0
?
Adam/dense_102/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*(
shared_nameAdam/dense_102/kernel/v
?
+Adam/dense_102/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_102/kernel/v*
_output_shapes

:@*
dtype0
?
Adam/dense_102/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_102/bias/v
{
)Adam/dense_102/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_102/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_103/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_103/kernel/v
?
+Adam/dense_103/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_103/kernel/v*
_output_shapes

:*
dtype0
?
Adam/dense_103/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_103/bias/v
{
)Adam/dense_103/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_103/bias/v*
_output_shapes
:*
dtype0
?
=Adam/token_and_position_embedding_9/embedding_20/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@@*N
shared_name?=Adam/token_and_position_embedding_9/embedding_20/embeddings/v
?
QAdam/token_and_position_embedding_9/embedding_20/embeddings/v/Read/ReadVariableOpReadVariableOp=Adam/token_and_position_embedding_9/embedding_20/embeddings/v*
_output_shapes
:	?@@*
dtype0
?
=Adam/token_and_position_embedding_9/embedding_21/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d@*N
shared_name?=Adam/token_and_position_embedding_9/embedding_21/embeddings/v
?
QAdam/token_and_position_embedding_9/embedding_21/embeddings/v/Read/ReadVariableOpReadVariableOp=Adam/token_and_position_embedding_9/embedding_21/embeddings/v*
_output_shapes

:d@*
dtype0
?
AAdam/transformer_block_8/multi_head_attention_8/dense_96/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*R
shared_nameCAAdam/transformer_block_8/multi_head_attention_8/dense_96/kernel/v
?
UAdam/transformer_block_8/multi_head_attention_8/dense_96/kernel/v/Read/ReadVariableOpReadVariableOpAAdam/transformer_block_8/multi_head_attention_8/dense_96/kernel/v*
_output_shapes

:@@*
dtype0
?
?Adam/transformer_block_8/multi_head_attention_8/dense_96/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*P
shared_nameA?Adam/transformer_block_8/multi_head_attention_8/dense_96/bias/v
?
SAdam/transformer_block_8/multi_head_attention_8/dense_96/bias/v/Read/ReadVariableOpReadVariableOp?Adam/transformer_block_8/multi_head_attention_8/dense_96/bias/v*
_output_shapes
:@*
dtype0
?
AAdam/transformer_block_8/multi_head_attention_8/dense_97/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*R
shared_nameCAAdam/transformer_block_8/multi_head_attention_8/dense_97/kernel/v
?
UAdam/transformer_block_8/multi_head_attention_8/dense_97/kernel/v/Read/ReadVariableOpReadVariableOpAAdam/transformer_block_8/multi_head_attention_8/dense_97/kernel/v*
_output_shapes

:@@*
dtype0
?
?Adam/transformer_block_8/multi_head_attention_8/dense_97/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*P
shared_nameA?Adam/transformer_block_8/multi_head_attention_8/dense_97/bias/v
?
SAdam/transformer_block_8/multi_head_attention_8/dense_97/bias/v/Read/ReadVariableOpReadVariableOp?Adam/transformer_block_8/multi_head_attention_8/dense_97/bias/v*
_output_shapes
:@*
dtype0
?
AAdam/transformer_block_8/multi_head_attention_8/dense_98/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*R
shared_nameCAAdam/transformer_block_8/multi_head_attention_8/dense_98/kernel/v
?
UAdam/transformer_block_8/multi_head_attention_8/dense_98/kernel/v/Read/ReadVariableOpReadVariableOpAAdam/transformer_block_8/multi_head_attention_8/dense_98/kernel/v*
_output_shapes

:@@*
dtype0
?
?Adam/transformer_block_8/multi_head_attention_8/dense_98/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*P
shared_nameA?Adam/transformer_block_8/multi_head_attention_8/dense_98/bias/v
?
SAdam/transformer_block_8/multi_head_attention_8/dense_98/bias/v/Read/ReadVariableOpReadVariableOp?Adam/transformer_block_8/multi_head_attention_8/dense_98/bias/v*
_output_shapes
:@*
dtype0
?
AAdam/transformer_block_8/multi_head_attention_8/dense_99/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*R
shared_nameCAAdam/transformer_block_8/multi_head_attention_8/dense_99/kernel/v
?
UAdam/transformer_block_8/multi_head_attention_8/dense_99/kernel/v/Read/ReadVariableOpReadVariableOpAAdam/transformer_block_8/multi_head_attention_8/dense_99/kernel/v*
_output_shapes

:@@*
dtype0
?
?Adam/transformer_block_8/multi_head_attention_8/dense_99/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*P
shared_nameA?Adam/transformer_block_8/multi_head_attention_8/dense_99/bias/v
?
SAdam/transformer_block_8/multi_head_attention_8/dense_99/bias/v/Read/ReadVariableOpReadVariableOp?Adam/transformer_block_8/multi_head_attention_8/dense_99/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_100/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *(
shared_nameAdam/dense_100/kernel/v
?
+Adam/dense_100/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_100/kernel/v*
_output_shapes

:@ *
dtype0
?
Adam/dense_100/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/dense_100/bias/v
{
)Adam/dense_100/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_100/bias/v*
_output_shapes
: *
dtype0
?
Adam/dense_101/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*(
shared_nameAdam/dense_101/kernel/v
?
+Adam/dense_101/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_101/kernel/v*
_output_shapes

: @*
dtype0
?
Adam/dense_101/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/dense_101/bias/v
{
)Adam/dense_101/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_101/bias/v*
_output_shapes
:@*
dtype0
?
7Adam/transformer_block_8/layer_normalization_26/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*H
shared_name97Adam/transformer_block_8/layer_normalization_26/gamma/v
?
KAdam/transformer_block_8/layer_normalization_26/gamma/v/Read/ReadVariableOpReadVariableOp7Adam/transformer_block_8/layer_normalization_26/gamma/v*
_output_shapes
:@*
dtype0
?
6Adam/transformer_block_8/layer_normalization_26/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*G
shared_name86Adam/transformer_block_8/layer_normalization_26/beta/v
?
JAdam/transformer_block_8/layer_normalization_26/beta/v/Read/ReadVariableOpReadVariableOp6Adam/transformer_block_8/layer_normalization_26/beta/v*
_output_shapes
:@*
dtype0
?
7Adam/transformer_block_8/layer_normalization_27/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*H
shared_name97Adam/transformer_block_8/layer_normalization_27/gamma/v
?
KAdam/transformer_block_8/layer_normalization_27/gamma/v/Read/ReadVariableOpReadVariableOp7Adam/transformer_block_8/layer_normalization_27/gamma/v*
_output_shapes
:@*
dtype0
?
6Adam/transformer_block_8/layer_normalization_27/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*G
shared_name86Adam/transformer_block_8/layer_normalization_27/beta/v
?
JAdam/transformer_block_8/layer_normalization_27/beta/v/Read/ReadVariableOpReadVariableOp6Adam/transformer_block_8/layer_normalization_27/beta/v*
_output_shapes
:@*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
		optimizer

trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
n
	token_emb
pos_emb
trainable_variables
regularization_losses
	variables
	keras_api
?
att
ffn

layernorm1

layernorm2
dropout1
dropout2
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
 regularization_losses
!	variables
"	keras_api
R
#trainable_variables
$regularization_losses
%	variables
&	keras_api
h

'kernel
(bias
)trainable_variables
*regularization_losses
+	variables
,	keras_api
R
-trainable_variables
.regularization_losses
/	variables
0	keras_api
h

1kernel
2bias
3trainable_variables
4regularization_losses
5	variables
6	keras_api
?
7iter

8beta_1

9beta_2
	:decay
;learning_rate'm?(m?1m?2m?<m?=m?>m??m?@m?Am?Bm?Cm?Dm?Em?Fm?Gm?Hm?Im?Jm?Km?Lm?Mm?'v?(v?1v?2v?<v?=v?>v??v?@v?Av?Bv?Cv?Dv?Ev?Fv?Gv?Hv?Iv?Jv?Kv?Lv?Mv?
?
<0
=1
>2
?3
@4
A5
B6
C7
D8
E9
F10
G11
H12
I13
J14
K15
L16
M17
'18
(19
120
221
 
?
<0
=1
>2
?3
@4
A5
B6
C7
D8
E9
F10
G11
H12
I13
J14
K15
L16
M17
'18
(19
120
221
?
Nlayer_regularization_losses

Olayers

trainable_variables
regularization_losses
	variables
Pmetrics
Qnon_trainable_variables
Rlayer_metrics
 
b
<
embeddings
Strainable_variables
Tregularization_losses
U	variables
V	keras_api
b
=
embeddings
Wtrainable_variables
Xregularization_losses
Y	variables
Z	keras_api

<0
=1
 

<0
=1
?
[layer_regularization_losses

\layers
trainable_variables
regularization_losses
	variables
]metrics
^non_trainable_variables
_layer_metrics
?
`query_dense
a	key_dense
bvalue_dense
	cdense
dtrainable_variables
eregularization_losses
f	variables
g	keras_api
?
hlayer_with_weights-0
hlayer-0
ilayer_with_weights-1
ilayer-1
jtrainable_variables
kregularization_losses
l	variables
m	keras_api
q
naxis
	Jgamma
Kbeta
otrainable_variables
pregularization_losses
q	variables
r	keras_api
q
saxis
	Lgamma
Mbeta
ttrainable_variables
uregularization_losses
v	variables
w	keras_api
R
xtrainable_variables
yregularization_losses
z	variables
{	keras_api
R
|trainable_variables
}regularization_losses
~	variables
	keras_api
v
>0
?1
@2
A3
B4
C5
D6
E7
F8
G9
H10
I11
J12
K13
L14
M15
 
v
>0
?1
@2
A3
B4
C5
D6
E7
F8
G9
H10
I11
J12
K13
L14
M15
?
 ?layer_regularization_losses
?layers
trainable_variables
regularization_losses
	variables
?metrics
?non_trainable_variables
?layer_metrics
 
 
 
?
 ?layer_regularization_losses
?layers
trainable_variables
 regularization_losses
!	variables
?metrics
?non_trainable_variables
?layer_metrics
 
 
 
?
 ?layer_regularization_losses
?layers
#trainable_variables
$regularization_losses
%	variables
?metrics
?non_trainable_variables
?layer_metrics
\Z
VARIABLE_VALUEdense_102/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_102/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

'0
(1
 

'0
(1
?
 ?layer_regularization_losses
?layers
)trainable_variables
*regularization_losses
+	variables
?metrics
?non_trainable_variables
?layer_metrics
 
 
 
?
 ?layer_regularization_losses
?layers
-trainable_variables
.regularization_losses
/	variables
?metrics
?non_trainable_variables
?layer_metrics
\Z
VARIABLE_VALUEdense_103/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_103/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

10
21
 

10
21
?
 ?layer_regularization_losses
?layers
3trainable_variables
4regularization_losses
5	variables
?metrics
?non_trainable_variables
?layer_metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE6token_and_position_embedding_9/embedding_20/embeddings0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE6token_and_position_embedding_9/embedding_21/embeddings0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUE:transformer_block_8/multi_head_attention_8/dense_96/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE8transformer_block_8/multi_head_attention_8/dense_96/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUE:transformer_block_8/multi_head_attention_8/dense_97/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE8transformer_block_8/multi_head_attention_8/dense_97/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUE:transformer_block_8/multi_head_attention_8/dense_98/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE8transformer_block_8/multi_head_attention_8/dense_98/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUE:transformer_block_8/multi_head_attention_8/dense_99/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE8transformer_block_8/multi_head_attention_8/dense_99/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_100/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_100/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_101/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_101/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE0transformer_block_8/layer_normalization_26/gamma1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE/transformer_block_8/layer_normalization_26/beta1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE0transformer_block_8/layer_normalization_27/gamma1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE/transformer_block_8/layer_normalization_27/beta1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
 
8
0
1
2
3
4
5
6
7

?0
?1
 
 

<0
 

<0
?
 ?layer_regularization_losses
?layers
Strainable_variables
Tregularization_losses
U	variables
?metrics
?non_trainable_variables
?layer_metrics

=0
 

=0
?
 ?layer_regularization_losses
?layers
Wtrainable_variables
Xregularization_losses
Y	variables
?metrics
?non_trainable_variables
?layer_metrics
 

0
1
 
 
 
l

>kernel
?bias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
l

@kernel
Abias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
l

Bkernel
Cbias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
l

Dkernel
Ebias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
8
>0
?1
@2
A3
B4
C5
D6
E7
 
8
>0
?1
@2
A3
B4
C5
D6
E7
?
 ?layer_regularization_losses
?layers
dtrainable_variables
eregularization_losses
f	variables
?metrics
?non_trainable_variables
?layer_metrics
l

Fkernel
Gbias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
l

Hkernel
Ibias
?trainable_variables
?regularization_losses
?	variables
?	keras_api

F0
G1
H2
I3
 

F0
G1
H2
I3
?
 ?layer_regularization_losses
?layers
jtrainable_variables
kregularization_losses
l	variables
?metrics
?non_trainable_variables
?layer_metrics
 

J0
K1
 

J0
K1
?
 ?layer_regularization_losses
?layers
otrainable_variables
pregularization_losses
q	variables
?metrics
?non_trainable_variables
?layer_metrics
 

L0
M1
 

L0
M1
?
 ?layer_regularization_losses
?layers
ttrainable_variables
uregularization_losses
v	variables
?metrics
?non_trainable_variables
?layer_metrics
 
 
 
?
 ?layer_regularization_losses
?layers
xtrainable_variables
yregularization_losses
z	variables
?metrics
?non_trainable_variables
?layer_metrics
 
 
 
?
 ?layer_regularization_losses
?layers
|trainable_variables
}regularization_losses
~	variables
?metrics
?non_trainable_variables
?layer_metrics
 
*
0
1
2
3
4
5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
 
 
 
 
 
 
 
 
 
 

>0
?1
 

>0
?1
?
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
?	variables
?metrics
?non_trainable_variables
?layer_metrics

@0
A1
 

@0
A1
?
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
?	variables
?metrics
?non_trainable_variables
?layer_metrics

B0
C1
 

B0
C1
?
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
?	variables
?metrics
?non_trainable_variables
?layer_metrics

D0
E1
 

D0
E1
?
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
?	variables
?metrics
?non_trainable_variables
?layer_metrics
 

`0
a1
b2
c3
 
 
 

F0
G1
 

F0
G1
?
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
?	variables
?metrics
?non_trainable_variables
?layer_metrics

H0
I1
 

H0
I1
?
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
?	variables
?metrics
?non_trainable_variables
?layer_metrics
 

h0
i1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
}
VARIABLE_VALUEAdam/dense_102/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_102/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_103/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_103/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE=Adam/token_and_position_embedding_9/embedding_20/embeddings/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE=Adam/token_and_position_embedding_9/embedding_21/embeddings/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAAdam/transformer_block_8/multi_head_attention_8/dense_96/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE?Adam/transformer_block_8/multi_head_attention_8/dense_96/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAAdam/transformer_block_8/multi_head_attention_8/dense_97/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE?Adam/transformer_block_8/multi_head_attention_8/dense_97/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAAdam/transformer_block_8/multi_head_attention_8/dense_98/kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE?Adam/transformer_block_8/multi_head_attention_8/dense_98/bias/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAAdam/transformer_block_8/multi_head_attention_8/dense_99/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE?Adam/transformer_block_8/multi_head_attention_8/dense_99/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_100/kernel/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_100/bias/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_101/kernel/mMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_101/bias/mMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE7Adam/transformer_block_8/layer_normalization_26/gamma/mMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/transformer_block_8/layer_normalization_26/beta/mMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE7Adam/transformer_block_8/layer_normalization_27/gamma/mMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/transformer_block_8/layer_normalization_27/beta/mMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_102/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_102/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_103/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_103/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE=Adam/token_and_position_embedding_9/embedding_20/embeddings/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE=Adam/token_and_position_embedding_9/embedding_21/embeddings/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAAdam/transformer_block_8/multi_head_attention_8/dense_96/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE?Adam/transformer_block_8/multi_head_attention_8/dense_96/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAAdam/transformer_block_8/multi_head_attention_8/dense_97/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE?Adam/transformer_block_8/multi_head_attention_8/dense_97/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAAdam/transformer_block_8/multi_head_attention_8/dense_98/kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE?Adam/transformer_block_8/multi_head_attention_8/dense_98/bias/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAAdam/transformer_block_8/multi_head_attention_8/dense_99/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE?Adam/transformer_block_8/multi_head_attention_8/dense_99/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_100/kernel/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_100/bias/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_101/kernel/vMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_101/bias/vMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE7Adam/transformer_block_8/layer_normalization_26/gamma/vMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/transformer_block_8/layer_normalization_26/beta/vMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE7Adam/transformer_block_8/layer_normalization_27/gamma/vMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/transformer_block_8/layer_normalization_27/beta/vMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{
serving_default_input_11Placeholder*'
_output_shapes
:?????????d*
dtype0*
shape:?????????d
?	
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_116token_and_position_embedding_9/embedding_21/embeddings6token_and_position_embedding_9/embedding_20/embeddings:transformer_block_8/multi_head_attention_8/dense_96/kernel8transformer_block_8/multi_head_attention_8/dense_96/bias:transformer_block_8/multi_head_attention_8/dense_97/kernel8transformer_block_8/multi_head_attention_8/dense_97/bias:transformer_block_8/multi_head_attention_8/dense_98/kernel8transformer_block_8/multi_head_attention_8/dense_98/bias:transformer_block_8/multi_head_attention_8/dense_99/kernel8transformer_block_8/multi_head_attention_8/dense_99/bias0transformer_block_8/layer_normalization_26/gamma/transformer_block_8/layer_normalization_26/betadense_100/kerneldense_100/biasdense_101/kerneldense_101/bias0transformer_block_8/layer_normalization_27/gamma/transformer_block_8/layer_normalization_27/betadense_102/kerneldense_102/biasdense_103/kerneldense_103/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_92229
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?'
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_102/kernel/Read/ReadVariableOp"dense_102/bias/Read/ReadVariableOp$dense_103/kernel/Read/ReadVariableOp"dense_103/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpJtoken_and_position_embedding_9/embedding_20/embeddings/Read/ReadVariableOpJtoken_and_position_embedding_9/embedding_21/embeddings/Read/ReadVariableOpNtransformer_block_8/multi_head_attention_8/dense_96/kernel/Read/ReadVariableOpLtransformer_block_8/multi_head_attention_8/dense_96/bias/Read/ReadVariableOpNtransformer_block_8/multi_head_attention_8/dense_97/kernel/Read/ReadVariableOpLtransformer_block_8/multi_head_attention_8/dense_97/bias/Read/ReadVariableOpNtransformer_block_8/multi_head_attention_8/dense_98/kernel/Read/ReadVariableOpLtransformer_block_8/multi_head_attention_8/dense_98/bias/Read/ReadVariableOpNtransformer_block_8/multi_head_attention_8/dense_99/kernel/Read/ReadVariableOpLtransformer_block_8/multi_head_attention_8/dense_99/bias/Read/ReadVariableOp$dense_100/kernel/Read/ReadVariableOp"dense_100/bias/Read/ReadVariableOp$dense_101/kernel/Read/ReadVariableOp"dense_101/bias/Read/ReadVariableOpDtransformer_block_8/layer_normalization_26/gamma/Read/ReadVariableOpCtransformer_block_8/layer_normalization_26/beta/Read/ReadVariableOpDtransformer_block_8/layer_normalization_27/gamma/Read/ReadVariableOpCtransformer_block_8/layer_normalization_27/beta/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/dense_102/kernel/m/Read/ReadVariableOp)Adam/dense_102/bias/m/Read/ReadVariableOp+Adam/dense_103/kernel/m/Read/ReadVariableOp)Adam/dense_103/bias/m/Read/ReadVariableOpQAdam/token_and_position_embedding_9/embedding_20/embeddings/m/Read/ReadVariableOpQAdam/token_and_position_embedding_9/embedding_21/embeddings/m/Read/ReadVariableOpUAdam/transformer_block_8/multi_head_attention_8/dense_96/kernel/m/Read/ReadVariableOpSAdam/transformer_block_8/multi_head_attention_8/dense_96/bias/m/Read/ReadVariableOpUAdam/transformer_block_8/multi_head_attention_8/dense_97/kernel/m/Read/ReadVariableOpSAdam/transformer_block_8/multi_head_attention_8/dense_97/bias/m/Read/ReadVariableOpUAdam/transformer_block_8/multi_head_attention_8/dense_98/kernel/m/Read/ReadVariableOpSAdam/transformer_block_8/multi_head_attention_8/dense_98/bias/m/Read/ReadVariableOpUAdam/transformer_block_8/multi_head_attention_8/dense_99/kernel/m/Read/ReadVariableOpSAdam/transformer_block_8/multi_head_attention_8/dense_99/bias/m/Read/ReadVariableOp+Adam/dense_100/kernel/m/Read/ReadVariableOp)Adam/dense_100/bias/m/Read/ReadVariableOp+Adam/dense_101/kernel/m/Read/ReadVariableOp)Adam/dense_101/bias/m/Read/ReadVariableOpKAdam/transformer_block_8/layer_normalization_26/gamma/m/Read/ReadVariableOpJAdam/transformer_block_8/layer_normalization_26/beta/m/Read/ReadVariableOpKAdam/transformer_block_8/layer_normalization_27/gamma/m/Read/ReadVariableOpJAdam/transformer_block_8/layer_normalization_27/beta/m/Read/ReadVariableOp+Adam/dense_102/kernel/v/Read/ReadVariableOp)Adam/dense_102/bias/v/Read/ReadVariableOp+Adam/dense_103/kernel/v/Read/ReadVariableOp)Adam/dense_103/bias/v/Read/ReadVariableOpQAdam/token_and_position_embedding_9/embedding_20/embeddings/v/Read/ReadVariableOpQAdam/token_and_position_embedding_9/embedding_21/embeddings/v/Read/ReadVariableOpUAdam/transformer_block_8/multi_head_attention_8/dense_96/kernel/v/Read/ReadVariableOpSAdam/transformer_block_8/multi_head_attention_8/dense_96/bias/v/Read/ReadVariableOpUAdam/transformer_block_8/multi_head_attention_8/dense_97/kernel/v/Read/ReadVariableOpSAdam/transformer_block_8/multi_head_attention_8/dense_97/bias/v/Read/ReadVariableOpUAdam/transformer_block_8/multi_head_attention_8/dense_98/kernel/v/Read/ReadVariableOpSAdam/transformer_block_8/multi_head_attention_8/dense_98/bias/v/Read/ReadVariableOpUAdam/transformer_block_8/multi_head_attention_8/dense_99/kernel/v/Read/ReadVariableOpSAdam/transformer_block_8/multi_head_attention_8/dense_99/bias/v/Read/ReadVariableOp+Adam/dense_100/kernel/v/Read/ReadVariableOp)Adam/dense_100/bias/v/Read/ReadVariableOp+Adam/dense_101/kernel/v/Read/ReadVariableOp)Adam/dense_101/bias/v/Read/ReadVariableOpKAdam/transformer_block_8/layer_normalization_26/gamma/v/Read/ReadVariableOpJAdam/transformer_block_8/layer_normalization_26/beta/v/Read/ReadVariableOpKAdam/transformer_block_8/layer_normalization_27/gamma/v/Read/ReadVariableOpJAdam/transformer_block_8/layer_normalization_27/beta/v/Read/ReadVariableOpConst*X
TinQ
O2M	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_94111
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_102/kerneldense_102/biasdense_103/kerneldense_103/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate6token_and_position_embedding_9/embedding_20/embeddings6token_and_position_embedding_9/embedding_21/embeddings:transformer_block_8/multi_head_attention_8/dense_96/kernel8transformer_block_8/multi_head_attention_8/dense_96/bias:transformer_block_8/multi_head_attention_8/dense_97/kernel8transformer_block_8/multi_head_attention_8/dense_97/bias:transformer_block_8/multi_head_attention_8/dense_98/kernel8transformer_block_8/multi_head_attention_8/dense_98/bias:transformer_block_8/multi_head_attention_8/dense_99/kernel8transformer_block_8/multi_head_attention_8/dense_99/biasdense_100/kerneldense_100/biasdense_101/kerneldense_101/bias0transformer_block_8/layer_normalization_26/gamma/transformer_block_8/layer_normalization_26/beta0transformer_block_8/layer_normalization_27/gamma/transformer_block_8/layer_normalization_27/betatotalcounttotal_1count_1Adam/dense_102/kernel/mAdam/dense_102/bias/mAdam/dense_103/kernel/mAdam/dense_103/bias/m=Adam/token_and_position_embedding_9/embedding_20/embeddings/m=Adam/token_and_position_embedding_9/embedding_21/embeddings/mAAdam/transformer_block_8/multi_head_attention_8/dense_96/kernel/m?Adam/transformer_block_8/multi_head_attention_8/dense_96/bias/mAAdam/transformer_block_8/multi_head_attention_8/dense_97/kernel/m?Adam/transformer_block_8/multi_head_attention_8/dense_97/bias/mAAdam/transformer_block_8/multi_head_attention_8/dense_98/kernel/m?Adam/transformer_block_8/multi_head_attention_8/dense_98/bias/mAAdam/transformer_block_8/multi_head_attention_8/dense_99/kernel/m?Adam/transformer_block_8/multi_head_attention_8/dense_99/bias/mAdam/dense_100/kernel/mAdam/dense_100/bias/mAdam/dense_101/kernel/mAdam/dense_101/bias/m7Adam/transformer_block_8/layer_normalization_26/gamma/m6Adam/transformer_block_8/layer_normalization_26/beta/m7Adam/transformer_block_8/layer_normalization_27/gamma/m6Adam/transformer_block_8/layer_normalization_27/beta/mAdam/dense_102/kernel/vAdam/dense_102/bias/vAdam/dense_103/kernel/vAdam/dense_103/bias/v=Adam/token_and_position_embedding_9/embedding_20/embeddings/v=Adam/token_and_position_embedding_9/embedding_21/embeddings/vAAdam/transformer_block_8/multi_head_attention_8/dense_96/kernel/v?Adam/transformer_block_8/multi_head_attention_8/dense_96/bias/vAAdam/transformer_block_8/multi_head_attention_8/dense_97/kernel/v?Adam/transformer_block_8/multi_head_attention_8/dense_97/bias/vAAdam/transformer_block_8/multi_head_attention_8/dense_98/kernel/v?Adam/transformer_block_8/multi_head_attention_8/dense_98/bias/vAAdam/transformer_block_8/multi_head_attention_8/dense_99/kernel/v?Adam/transformer_block_8/multi_head_attention_8/dense_99/bias/vAdam/dense_100/kernel/vAdam/dense_100/bias/vAdam/dense_101/kernel/vAdam/dense_101/bias/v7Adam/transformer_block_8/layer_normalization_26/gamma/v6Adam/transformer_block_8/layer_normalization_26/beta/v7Adam/transformer_block_8/layer_normalization_27/gamma/v6Adam/transformer_block_8/layer_normalization_27/beta/v*W
TinP
N2L*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_94346??(
?
q
U__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_91348

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indiceso
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d@:S O
+
_output_shapes
:?????????d@
 
_user_specified_nameinputs
?
c
E__inference_dropout_44_layer_call_and_return_conditional_losses_91355

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
q
U__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_91014

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
c
*__inference_dropout_45_layer_call_fn_93607

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_45_layer_call_and_return_conditional_losses_914762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
? 
?
D__inference_dense_100_layer_call_and_return_conditional_losses_93824

inputs3
!tensordot_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????d@2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????d 2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d 2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????d 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????d 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????d@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:?????????d@
 
_user_specified_nameinputs
? 
?
D__inference_dense_100_layer_call_and_return_conditional_losses_90849

inputs3
!tensordot_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????d@2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????d 2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d 2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????d 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????d 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????d@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:?????????d@
 
_user_specified_nameinputs
?
?
'__inference_model_8_layer_call_fn_92278

inputs
unknown:d@
	unknown_0:	?@@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@ 

unknown_12: 

unknown_13: @

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:

unknown_19:

unknown_20:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_8_layer_call_and_return_conditional_losses_913992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????d: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
??
?
N__inference_transformer_block_8_layer_call_and_return_conditional_losses_91815

inputsS
Amulti_head_attention_8_dense_96_tensordot_readvariableop_resource:@@M
?multi_head_attention_8_dense_96_biasadd_readvariableop_resource:@S
Amulti_head_attention_8_dense_97_tensordot_readvariableop_resource:@@M
?multi_head_attention_8_dense_97_biasadd_readvariableop_resource:@S
Amulti_head_attention_8_dense_98_tensordot_readvariableop_resource:@@M
?multi_head_attention_8_dense_98_biasadd_readvariableop_resource:@S
Amulti_head_attention_8_dense_99_tensordot_readvariableop_resource:@@M
?multi_head_attention_8_dense_99_biasadd_readvariableop_resource:@J
<layer_normalization_26_batchnorm_mul_readvariableop_resource:@F
8layer_normalization_26_batchnorm_readvariableop_resource:@J
8sequential_8_dense_100_tensordot_readvariableop_resource:@ D
6sequential_8_dense_100_biasadd_readvariableop_resource: J
8sequential_8_dense_101_tensordot_readvariableop_resource: @D
6sequential_8_dense_101_biasadd_readvariableop_resource:@J
<layer_normalization_27_batchnorm_mul_readvariableop_resource:@F
8layer_normalization_27_batchnorm_readvariableop_resource:@
identity??/layer_normalization_26/batchnorm/ReadVariableOp?3layer_normalization_26/batchnorm/mul/ReadVariableOp?/layer_normalization_27/batchnorm/ReadVariableOp?3layer_normalization_27/batchnorm/mul/ReadVariableOp?6multi_head_attention_8/dense_96/BiasAdd/ReadVariableOp?8multi_head_attention_8/dense_96/Tensordot/ReadVariableOp?6multi_head_attention_8/dense_97/BiasAdd/ReadVariableOp?8multi_head_attention_8/dense_97/Tensordot/ReadVariableOp?6multi_head_attention_8/dense_98/BiasAdd/ReadVariableOp?8multi_head_attention_8/dense_98/Tensordot/ReadVariableOp?6multi_head_attention_8/dense_99/BiasAdd/ReadVariableOp?8multi_head_attention_8/dense_99/Tensordot/ReadVariableOp?-sequential_8/dense_100/BiasAdd/ReadVariableOp?/sequential_8/dense_100/Tensordot/ReadVariableOp?-sequential_8/dense_101/BiasAdd/ReadVariableOp?/sequential_8/dense_101/Tensordot/ReadVariableOpr
multi_head_attention_8/ShapeShapeinputs*
T0*
_output_shapes
:2
multi_head_attention_8/Shape?
*multi_head_attention_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*multi_head_attention_8/strided_slice/stack?
,multi_head_attention_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,multi_head_attention_8/strided_slice/stack_1?
,multi_head_attention_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,multi_head_attention_8/strided_slice/stack_2?
$multi_head_attention_8/strided_sliceStridedSlice%multi_head_attention_8/Shape:output:03multi_head_attention_8/strided_slice/stack:output:05multi_head_attention_8/strided_slice/stack_1:output:05multi_head_attention_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$multi_head_attention_8/strided_slice?
8multi_head_attention_8/dense_96/Tensordot/ReadVariableOpReadVariableOpAmulti_head_attention_8_dense_96_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02:
8multi_head_attention_8/dense_96/Tensordot/ReadVariableOp?
.multi_head_attention_8/dense_96/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:20
.multi_head_attention_8/dense_96/Tensordot/axes?
.multi_head_attention_8/dense_96/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       20
.multi_head_attention_8/dense_96/Tensordot/free?
/multi_head_attention_8/dense_96/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:21
/multi_head_attention_8/dense_96/Tensordot/Shape?
7multi_head_attention_8/dense_96/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_attention_8/dense_96/Tensordot/GatherV2/axis?
2multi_head_attention_8/dense_96/Tensordot/GatherV2GatherV28multi_head_attention_8/dense_96/Tensordot/Shape:output:07multi_head_attention_8/dense_96/Tensordot/free:output:0@multi_head_attention_8/dense_96/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:24
2multi_head_attention_8/dense_96/Tensordot/GatherV2?
9multi_head_attention_8/dense_96/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9multi_head_attention_8/dense_96/Tensordot/GatherV2_1/axis?
4multi_head_attention_8/dense_96/Tensordot/GatherV2_1GatherV28multi_head_attention_8/dense_96/Tensordot/Shape:output:07multi_head_attention_8/dense_96/Tensordot/axes:output:0Bmulti_head_attention_8/dense_96/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4multi_head_attention_8/dense_96/Tensordot/GatherV2_1?
/multi_head_attention_8/dense_96/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/multi_head_attention_8/dense_96/Tensordot/Const?
.multi_head_attention_8/dense_96/Tensordot/ProdProd;multi_head_attention_8/dense_96/Tensordot/GatherV2:output:08multi_head_attention_8/dense_96/Tensordot/Const:output:0*
T0*
_output_shapes
: 20
.multi_head_attention_8/dense_96/Tensordot/Prod?
1multi_head_attention_8/dense_96/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1multi_head_attention_8/dense_96/Tensordot/Const_1?
0multi_head_attention_8/dense_96/Tensordot/Prod_1Prod=multi_head_attention_8/dense_96/Tensordot/GatherV2_1:output:0:multi_head_attention_8/dense_96/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 22
0multi_head_attention_8/dense_96/Tensordot/Prod_1?
5multi_head_attention_8/dense_96/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5multi_head_attention_8/dense_96/Tensordot/concat/axis?
0multi_head_attention_8/dense_96/Tensordot/concatConcatV27multi_head_attention_8/dense_96/Tensordot/free:output:07multi_head_attention_8/dense_96/Tensordot/axes:output:0>multi_head_attention_8/dense_96/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:22
0multi_head_attention_8/dense_96/Tensordot/concat?
/multi_head_attention_8/dense_96/Tensordot/stackPack7multi_head_attention_8/dense_96/Tensordot/Prod:output:09multi_head_attention_8/dense_96/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:21
/multi_head_attention_8/dense_96/Tensordot/stack?
3multi_head_attention_8/dense_96/Tensordot/transpose	Transposeinputs9multi_head_attention_8/dense_96/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????d@25
3multi_head_attention_8/dense_96/Tensordot/transpose?
1multi_head_attention_8/dense_96/Tensordot/ReshapeReshape7multi_head_attention_8/dense_96/Tensordot/transpose:y:08multi_head_attention_8/dense_96/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????23
1multi_head_attention_8/dense_96/Tensordot/Reshape?
0multi_head_attention_8/dense_96/Tensordot/MatMulMatMul:multi_head_attention_8/dense_96/Tensordot/Reshape:output:0@multi_head_attention_8/dense_96/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@22
0multi_head_attention_8/dense_96/Tensordot/MatMul?
1multi_head_attention_8/dense_96/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@23
1multi_head_attention_8/dense_96/Tensordot/Const_2?
7multi_head_attention_8/dense_96/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_attention_8/dense_96/Tensordot/concat_1/axis?
2multi_head_attention_8/dense_96/Tensordot/concat_1ConcatV2;multi_head_attention_8/dense_96/Tensordot/GatherV2:output:0:multi_head_attention_8/dense_96/Tensordot/Const_2:output:0@multi_head_attention_8/dense_96/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:24
2multi_head_attention_8/dense_96/Tensordot/concat_1?
)multi_head_attention_8/dense_96/TensordotReshape:multi_head_attention_8/dense_96/Tensordot/MatMul:product:0;multi_head_attention_8/dense_96/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????d@2+
)multi_head_attention_8/dense_96/Tensordot?
6multi_head_attention_8/dense_96/BiasAdd/ReadVariableOpReadVariableOp?multi_head_attention_8_dense_96_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6multi_head_attention_8/dense_96/BiasAdd/ReadVariableOp?
'multi_head_attention_8/dense_96/BiasAddBiasAdd2multi_head_attention_8/dense_96/Tensordot:output:0>multi_head_attention_8/dense_96/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@2)
'multi_head_attention_8/dense_96/BiasAdd?
8multi_head_attention_8/dense_97/Tensordot/ReadVariableOpReadVariableOpAmulti_head_attention_8_dense_97_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02:
8multi_head_attention_8/dense_97/Tensordot/ReadVariableOp?
.multi_head_attention_8/dense_97/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:20
.multi_head_attention_8/dense_97/Tensordot/axes?
.multi_head_attention_8/dense_97/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       20
.multi_head_attention_8/dense_97/Tensordot/free?
/multi_head_attention_8/dense_97/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:21
/multi_head_attention_8/dense_97/Tensordot/Shape?
7multi_head_attention_8/dense_97/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_attention_8/dense_97/Tensordot/GatherV2/axis?
2multi_head_attention_8/dense_97/Tensordot/GatherV2GatherV28multi_head_attention_8/dense_97/Tensordot/Shape:output:07multi_head_attention_8/dense_97/Tensordot/free:output:0@multi_head_attention_8/dense_97/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:24
2multi_head_attention_8/dense_97/Tensordot/GatherV2?
9multi_head_attention_8/dense_97/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9multi_head_attention_8/dense_97/Tensordot/GatherV2_1/axis?
4multi_head_attention_8/dense_97/Tensordot/GatherV2_1GatherV28multi_head_attention_8/dense_97/Tensordot/Shape:output:07multi_head_attention_8/dense_97/Tensordot/axes:output:0Bmulti_head_attention_8/dense_97/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4multi_head_attention_8/dense_97/Tensordot/GatherV2_1?
/multi_head_attention_8/dense_97/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/multi_head_attention_8/dense_97/Tensordot/Const?
.multi_head_attention_8/dense_97/Tensordot/ProdProd;multi_head_attention_8/dense_97/Tensordot/GatherV2:output:08multi_head_attention_8/dense_97/Tensordot/Const:output:0*
T0*
_output_shapes
: 20
.multi_head_attention_8/dense_97/Tensordot/Prod?
1multi_head_attention_8/dense_97/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1multi_head_attention_8/dense_97/Tensordot/Const_1?
0multi_head_attention_8/dense_97/Tensordot/Prod_1Prod=multi_head_attention_8/dense_97/Tensordot/GatherV2_1:output:0:multi_head_attention_8/dense_97/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 22
0multi_head_attention_8/dense_97/Tensordot/Prod_1?
5multi_head_attention_8/dense_97/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5multi_head_attention_8/dense_97/Tensordot/concat/axis?
0multi_head_attention_8/dense_97/Tensordot/concatConcatV27multi_head_attention_8/dense_97/Tensordot/free:output:07multi_head_attention_8/dense_97/Tensordot/axes:output:0>multi_head_attention_8/dense_97/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:22
0multi_head_attention_8/dense_97/Tensordot/concat?
/multi_head_attention_8/dense_97/Tensordot/stackPack7multi_head_attention_8/dense_97/Tensordot/Prod:output:09multi_head_attention_8/dense_97/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:21
/multi_head_attention_8/dense_97/Tensordot/stack?
3multi_head_attention_8/dense_97/Tensordot/transpose	Transposeinputs9multi_head_attention_8/dense_97/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????d@25
3multi_head_attention_8/dense_97/Tensordot/transpose?
1multi_head_attention_8/dense_97/Tensordot/ReshapeReshape7multi_head_attention_8/dense_97/Tensordot/transpose:y:08multi_head_attention_8/dense_97/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????23
1multi_head_attention_8/dense_97/Tensordot/Reshape?
0multi_head_attention_8/dense_97/Tensordot/MatMulMatMul:multi_head_attention_8/dense_97/Tensordot/Reshape:output:0@multi_head_attention_8/dense_97/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@22
0multi_head_attention_8/dense_97/Tensordot/MatMul?
1multi_head_attention_8/dense_97/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@23
1multi_head_attention_8/dense_97/Tensordot/Const_2?
7multi_head_attention_8/dense_97/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_attention_8/dense_97/Tensordot/concat_1/axis?
2multi_head_attention_8/dense_97/Tensordot/concat_1ConcatV2;multi_head_attention_8/dense_97/Tensordot/GatherV2:output:0:multi_head_attention_8/dense_97/Tensordot/Const_2:output:0@multi_head_attention_8/dense_97/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:24
2multi_head_attention_8/dense_97/Tensordot/concat_1?
)multi_head_attention_8/dense_97/TensordotReshape:multi_head_attention_8/dense_97/Tensordot/MatMul:product:0;multi_head_attention_8/dense_97/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????d@2+
)multi_head_attention_8/dense_97/Tensordot?
6multi_head_attention_8/dense_97/BiasAdd/ReadVariableOpReadVariableOp?multi_head_attention_8_dense_97_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6multi_head_attention_8/dense_97/BiasAdd/ReadVariableOp?
'multi_head_attention_8/dense_97/BiasAddBiasAdd2multi_head_attention_8/dense_97/Tensordot:output:0>multi_head_attention_8/dense_97/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@2)
'multi_head_attention_8/dense_97/BiasAdd?
8multi_head_attention_8/dense_98/Tensordot/ReadVariableOpReadVariableOpAmulti_head_attention_8_dense_98_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02:
8multi_head_attention_8/dense_98/Tensordot/ReadVariableOp?
.multi_head_attention_8/dense_98/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:20
.multi_head_attention_8/dense_98/Tensordot/axes?
.multi_head_attention_8/dense_98/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       20
.multi_head_attention_8/dense_98/Tensordot/free?
/multi_head_attention_8/dense_98/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:21
/multi_head_attention_8/dense_98/Tensordot/Shape?
7multi_head_attention_8/dense_98/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_attention_8/dense_98/Tensordot/GatherV2/axis?
2multi_head_attention_8/dense_98/Tensordot/GatherV2GatherV28multi_head_attention_8/dense_98/Tensordot/Shape:output:07multi_head_attention_8/dense_98/Tensordot/free:output:0@multi_head_attention_8/dense_98/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:24
2multi_head_attention_8/dense_98/Tensordot/GatherV2?
9multi_head_attention_8/dense_98/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9multi_head_attention_8/dense_98/Tensordot/GatherV2_1/axis?
4multi_head_attention_8/dense_98/Tensordot/GatherV2_1GatherV28multi_head_attention_8/dense_98/Tensordot/Shape:output:07multi_head_attention_8/dense_98/Tensordot/axes:output:0Bmulti_head_attention_8/dense_98/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4multi_head_attention_8/dense_98/Tensordot/GatherV2_1?
/multi_head_attention_8/dense_98/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/multi_head_attention_8/dense_98/Tensordot/Const?
.multi_head_attention_8/dense_98/Tensordot/ProdProd;multi_head_attention_8/dense_98/Tensordot/GatherV2:output:08multi_head_attention_8/dense_98/Tensordot/Const:output:0*
T0*
_output_shapes
: 20
.multi_head_attention_8/dense_98/Tensordot/Prod?
1multi_head_attention_8/dense_98/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1multi_head_attention_8/dense_98/Tensordot/Const_1?
0multi_head_attention_8/dense_98/Tensordot/Prod_1Prod=multi_head_attention_8/dense_98/Tensordot/GatherV2_1:output:0:multi_head_attention_8/dense_98/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 22
0multi_head_attention_8/dense_98/Tensordot/Prod_1?
5multi_head_attention_8/dense_98/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5multi_head_attention_8/dense_98/Tensordot/concat/axis?
0multi_head_attention_8/dense_98/Tensordot/concatConcatV27multi_head_attention_8/dense_98/Tensordot/free:output:07multi_head_attention_8/dense_98/Tensordot/axes:output:0>multi_head_attention_8/dense_98/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:22
0multi_head_attention_8/dense_98/Tensordot/concat?
/multi_head_attention_8/dense_98/Tensordot/stackPack7multi_head_attention_8/dense_98/Tensordot/Prod:output:09multi_head_attention_8/dense_98/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:21
/multi_head_attention_8/dense_98/Tensordot/stack?
3multi_head_attention_8/dense_98/Tensordot/transpose	Transposeinputs9multi_head_attention_8/dense_98/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????d@25
3multi_head_attention_8/dense_98/Tensordot/transpose?
1multi_head_attention_8/dense_98/Tensordot/ReshapeReshape7multi_head_attention_8/dense_98/Tensordot/transpose:y:08multi_head_attention_8/dense_98/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????23
1multi_head_attention_8/dense_98/Tensordot/Reshape?
0multi_head_attention_8/dense_98/Tensordot/MatMulMatMul:multi_head_attention_8/dense_98/Tensordot/Reshape:output:0@multi_head_attention_8/dense_98/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@22
0multi_head_attention_8/dense_98/Tensordot/MatMul?
1multi_head_attention_8/dense_98/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@23
1multi_head_attention_8/dense_98/Tensordot/Const_2?
7multi_head_attention_8/dense_98/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_attention_8/dense_98/Tensordot/concat_1/axis?
2multi_head_attention_8/dense_98/Tensordot/concat_1ConcatV2;multi_head_attention_8/dense_98/Tensordot/GatherV2:output:0:multi_head_attention_8/dense_98/Tensordot/Const_2:output:0@multi_head_attention_8/dense_98/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:24
2multi_head_attention_8/dense_98/Tensordot/concat_1?
)multi_head_attention_8/dense_98/TensordotReshape:multi_head_attention_8/dense_98/Tensordot/MatMul:product:0;multi_head_attention_8/dense_98/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????d@2+
)multi_head_attention_8/dense_98/Tensordot?
6multi_head_attention_8/dense_98/BiasAdd/ReadVariableOpReadVariableOp?multi_head_attention_8_dense_98_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6multi_head_attention_8/dense_98/BiasAdd/ReadVariableOp?
'multi_head_attention_8/dense_98/BiasAddBiasAdd2multi_head_attention_8/dense_98/Tensordot:output:0>multi_head_attention_8/dense_98/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@2)
'multi_head_attention_8/dense_98/BiasAdd?
&multi_head_attention_8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&multi_head_attention_8/Reshape/shape/1?
&multi_head_attention_8/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2(
&multi_head_attention_8/Reshape/shape/2?
&multi_head_attention_8/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2(
&multi_head_attention_8/Reshape/shape/3?
$multi_head_attention_8/Reshape/shapePack-multi_head_attention_8/strided_slice:output:0/multi_head_attention_8/Reshape/shape/1:output:0/multi_head_attention_8/Reshape/shape/2:output:0/multi_head_attention_8/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2&
$multi_head_attention_8/Reshape/shape?
multi_head_attention_8/ReshapeReshape0multi_head_attention_8/dense_96/BiasAdd:output:0-multi_head_attention_8/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2 
multi_head_attention_8/Reshape?
%multi_head_attention_8/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2'
%multi_head_attention_8/transpose/perm?
 multi_head_attention_8/transpose	Transpose'multi_head_attention_8/Reshape:output:0.multi_head_attention_8/transpose/perm:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2"
 multi_head_attention_8/transpose?
(multi_head_attention_8/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(multi_head_attention_8/Reshape_1/shape/1?
(multi_head_attention_8/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(multi_head_attention_8/Reshape_1/shape/2?
(multi_head_attention_8/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2*
(multi_head_attention_8/Reshape_1/shape/3?
&multi_head_attention_8/Reshape_1/shapePack-multi_head_attention_8/strided_slice:output:01multi_head_attention_8/Reshape_1/shape/1:output:01multi_head_attention_8/Reshape_1/shape/2:output:01multi_head_attention_8/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_attention_8/Reshape_1/shape?
 multi_head_attention_8/Reshape_1Reshape0multi_head_attention_8/dense_97/BiasAdd:output:0/multi_head_attention_8/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2"
 multi_head_attention_8/Reshape_1?
'multi_head_attention_8/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2)
'multi_head_attention_8/transpose_1/perm?
"multi_head_attention_8/transpose_1	Transpose)multi_head_attention_8/Reshape_1:output:00multi_head_attention_8/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2$
"multi_head_attention_8/transpose_1?
(multi_head_attention_8/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(multi_head_attention_8/Reshape_2/shape/1?
(multi_head_attention_8/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(multi_head_attention_8/Reshape_2/shape/2?
(multi_head_attention_8/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2*
(multi_head_attention_8/Reshape_2/shape/3?
&multi_head_attention_8/Reshape_2/shapePack-multi_head_attention_8/strided_slice:output:01multi_head_attention_8/Reshape_2/shape/1:output:01multi_head_attention_8/Reshape_2/shape/2:output:01multi_head_attention_8/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_attention_8/Reshape_2/shape?
 multi_head_attention_8/Reshape_2Reshape0multi_head_attention_8/dense_98/BiasAdd:output:0/multi_head_attention_8/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2"
 multi_head_attention_8/Reshape_2?
'multi_head_attention_8/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2)
'multi_head_attention_8/transpose_2/perm?
"multi_head_attention_8/transpose_2	Transpose)multi_head_attention_8/Reshape_2:output:00multi_head_attention_8/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2$
"multi_head_attention_8/transpose_2?
multi_head_attention_8/MatMulBatchMatMulV2$multi_head_attention_8/transpose:y:0&multi_head_attention_8/transpose_1:y:0*
T0*A
_output_shapes/
-:+???????????????????????????*
adj_y(2
multi_head_attention_8/MatMul?
multi_head_attention_8/Shape_1Shape&multi_head_attention_8/transpose_1:y:0*
T0*
_output_shapes
:2 
multi_head_attention_8/Shape_1?
,multi_head_attention_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2.
,multi_head_attention_8/strided_slice_1/stack?
.multi_head_attention_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 20
.multi_head_attention_8/strided_slice_1/stack_1?
.multi_head_attention_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.multi_head_attention_8/strided_slice_1/stack_2?
&multi_head_attention_8/strided_slice_1StridedSlice'multi_head_attention_8/Shape_1:output:05multi_head_attention_8/strided_slice_1/stack:output:07multi_head_attention_8/strided_slice_1/stack_1:output:07multi_head_attention_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&multi_head_attention_8/strided_slice_1?
multi_head_attention_8/CastCast/multi_head_attention_8/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
multi_head_attention_8/Cast?
multi_head_attention_8/SqrtSqrtmulti_head_attention_8/Cast:y:0*
T0*
_output_shapes
: 2
multi_head_attention_8/Sqrt?
multi_head_attention_8/truedivRealDiv&multi_head_attention_8/MatMul:output:0multi_head_attention_8/Sqrt:y:0*
T0*A
_output_shapes/
-:+???????????????????????????2 
multi_head_attention_8/truediv?
multi_head_attention_8/SoftmaxSoftmax"multi_head_attention_8/truediv:z:0*
T0*A
_output_shapes/
-:+???????????????????????????2 
multi_head_attention_8/Softmax?
multi_head_attention_8/MatMul_1BatchMatMulV2(multi_head_attention_8/Softmax:softmax:0&multi_head_attention_8/transpose_2:y:0*
T0*8
_output_shapes&
$:"?????????????????? 2!
multi_head_attention_8/MatMul_1?
'multi_head_attention_8/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             2)
'multi_head_attention_8/transpose_3/perm?
"multi_head_attention_8/transpose_3	Transpose(multi_head_attention_8/MatMul_1:output:00multi_head_attention_8/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2$
"multi_head_attention_8/transpose_3?
(multi_head_attention_8/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(multi_head_attention_8/Reshape_3/shape/1?
(multi_head_attention_8/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@2*
(multi_head_attention_8/Reshape_3/shape/2?
&multi_head_attention_8/Reshape_3/shapePack-multi_head_attention_8/strided_slice:output:01multi_head_attention_8/Reshape_3/shape/1:output:01multi_head_attention_8/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_attention_8/Reshape_3/shape?
 multi_head_attention_8/Reshape_3Reshape&multi_head_attention_8/transpose_3:y:0/multi_head_attention_8/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :??????????????????@2"
 multi_head_attention_8/Reshape_3?
8multi_head_attention_8/dense_99/Tensordot/ReadVariableOpReadVariableOpAmulti_head_attention_8_dense_99_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02:
8multi_head_attention_8/dense_99/Tensordot/ReadVariableOp?
.multi_head_attention_8/dense_99/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:20
.multi_head_attention_8/dense_99/Tensordot/axes?
.multi_head_attention_8/dense_99/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       20
.multi_head_attention_8/dense_99/Tensordot/free?
/multi_head_attention_8/dense_99/Tensordot/ShapeShape)multi_head_attention_8/Reshape_3:output:0*
T0*
_output_shapes
:21
/multi_head_attention_8/dense_99/Tensordot/Shape?
7multi_head_attention_8/dense_99/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_attention_8/dense_99/Tensordot/GatherV2/axis?
2multi_head_attention_8/dense_99/Tensordot/GatherV2GatherV28multi_head_attention_8/dense_99/Tensordot/Shape:output:07multi_head_attention_8/dense_99/Tensordot/free:output:0@multi_head_attention_8/dense_99/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:24
2multi_head_attention_8/dense_99/Tensordot/GatherV2?
9multi_head_attention_8/dense_99/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9multi_head_attention_8/dense_99/Tensordot/GatherV2_1/axis?
4multi_head_attention_8/dense_99/Tensordot/GatherV2_1GatherV28multi_head_attention_8/dense_99/Tensordot/Shape:output:07multi_head_attention_8/dense_99/Tensordot/axes:output:0Bmulti_head_attention_8/dense_99/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4multi_head_attention_8/dense_99/Tensordot/GatherV2_1?
/multi_head_attention_8/dense_99/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/multi_head_attention_8/dense_99/Tensordot/Const?
.multi_head_attention_8/dense_99/Tensordot/ProdProd;multi_head_attention_8/dense_99/Tensordot/GatherV2:output:08multi_head_attention_8/dense_99/Tensordot/Const:output:0*
T0*
_output_shapes
: 20
.multi_head_attention_8/dense_99/Tensordot/Prod?
1multi_head_attention_8/dense_99/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1multi_head_attention_8/dense_99/Tensordot/Const_1?
0multi_head_attention_8/dense_99/Tensordot/Prod_1Prod=multi_head_attention_8/dense_99/Tensordot/GatherV2_1:output:0:multi_head_attention_8/dense_99/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 22
0multi_head_attention_8/dense_99/Tensordot/Prod_1?
5multi_head_attention_8/dense_99/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5multi_head_attention_8/dense_99/Tensordot/concat/axis?
0multi_head_attention_8/dense_99/Tensordot/concatConcatV27multi_head_attention_8/dense_99/Tensordot/free:output:07multi_head_attention_8/dense_99/Tensordot/axes:output:0>multi_head_attention_8/dense_99/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:22
0multi_head_attention_8/dense_99/Tensordot/concat?
/multi_head_attention_8/dense_99/Tensordot/stackPack7multi_head_attention_8/dense_99/Tensordot/Prod:output:09multi_head_attention_8/dense_99/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:21
/multi_head_attention_8/dense_99/Tensordot/stack?
3multi_head_attention_8/dense_99/Tensordot/transpose	Transpose)multi_head_attention_8/Reshape_3:output:09multi_head_attention_8/dense_99/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :??????????????????@25
3multi_head_attention_8/dense_99/Tensordot/transpose?
1multi_head_attention_8/dense_99/Tensordot/ReshapeReshape7multi_head_attention_8/dense_99/Tensordot/transpose:y:08multi_head_attention_8/dense_99/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????23
1multi_head_attention_8/dense_99/Tensordot/Reshape?
0multi_head_attention_8/dense_99/Tensordot/MatMulMatMul:multi_head_attention_8/dense_99/Tensordot/Reshape:output:0@multi_head_attention_8/dense_99/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@22
0multi_head_attention_8/dense_99/Tensordot/MatMul?
1multi_head_attention_8/dense_99/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@23
1multi_head_attention_8/dense_99/Tensordot/Const_2?
7multi_head_attention_8/dense_99/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_attention_8/dense_99/Tensordot/concat_1/axis?
2multi_head_attention_8/dense_99/Tensordot/concat_1ConcatV2;multi_head_attention_8/dense_99/Tensordot/GatherV2:output:0:multi_head_attention_8/dense_99/Tensordot/Const_2:output:0@multi_head_attention_8/dense_99/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:24
2multi_head_attention_8/dense_99/Tensordot/concat_1?
)multi_head_attention_8/dense_99/TensordotReshape:multi_head_attention_8/dense_99/Tensordot/MatMul:product:0;multi_head_attention_8/dense_99/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :??????????????????@2+
)multi_head_attention_8/dense_99/Tensordot?
6multi_head_attention_8/dense_99/BiasAdd/ReadVariableOpReadVariableOp?multi_head_attention_8_dense_99_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6multi_head_attention_8/dense_99/BiasAdd/ReadVariableOp?
'multi_head_attention_8/dense_99/BiasAddBiasAdd2multi_head_attention_8/dense_99/Tensordot:output:0>multi_head_attention_8/dense_99/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????@2)
'multi_head_attention_8/dense_99/BiasAddy
dropout_42/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_42/dropout/Const?
dropout_42/dropout/MulMul0multi_head_attention_8/dense_99/BiasAdd:output:0!dropout_42/dropout/Const:output:0*
T0*4
_output_shapes"
 :??????????????????@2
dropout_42/dropout/Mul?
dropout_42/dropout/ShapeShape0multi_head_attention_8/dense_99/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_42/dropout/Shape?
/dropout_42/dropout/random_uniform/RandomUniformRandomUniform!dropout_42/dropout/Shape:output:0*
T0*4
_output_shapes"
 :??????????????????@*
dtype021
/dropout_42/dropout/random_uniform/RandomUniform?
!dropout_42/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2#
!dropout_42/dropout/GreaterEqual/y?
dropout_42/dropout/GreaterEqualGreaterEqual8dropout_42/dropout/random_uniform/RandomUniform:output:0*dropout_42/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????@2!
dropout_42/dropout/GreaterEqual?
dropout_42/dropout/CastCast#dropout_42/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????@2
dropout_42/dropout/Cast?
dropout_42/dropout/Mul_1Muldropout_42/dropout/Mul:z:0dropout_42/dropout/Cast:y:0*
T0*4
_output_shapes"
 :??????????????????@2
dropout_42/dropout/Mul_1o
addAddV2inputsdropout_42/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????d@2
add?
5layer_normalization_26/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_26/moments/mean/reduction_indices?
#layer_normalization_26/moments/meanMeanadd:z:0>layer_normalization_26/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????d*
	keep_dims(2%
#layer_normalization_26/moments/mean?
+layer_normalization_26/moments/StopGradientStopGradient,layer_normalization_26/moments/mean:output:0*
T0*+
_output_shapes
:?????????d2-
+layer_normalization_26/moments/StopGradient?
0layer_normalization_26/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_26/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????d@22
0layer_normalization_26/moments/SquaredDifference?
9layer_normalization_26/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_26/moments/variance/reduction_indices?
'layer_normalization_26/moments/varianceMean4layer_normalization_26/moments/SquaredDifference:z:0Blayer_normalization_26/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????d*
	keep_dims(2)
'layer_normalization_26/moments/variance?
&layer_normalization_26/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52(
&layer_normalization_26/batchnorm/add/y?
$layer_normalization_26/batchnorm/addAddV20layer_normalization_26/moments/variance:output:0/layer_normalization_26/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????d2&
$layer_normalization_26/batchnorm/add?
&layer_normalization_26/batchnorm/RsqrtRsqrt(layer_normalization_26/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????d2(
&layer_normalization_26/batchnorm/Rsqrt?
3layer_normalization_26/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_26_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype025
3layer_normalization_26/batchnorm/mul/ReadVariableOp?
$layer_normalization_26/batchnorm/mulMul*layer_normalization_26/batchnorm/Rsqrt:y:0;layer_normalization_26/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@2&
$layer_normalization_26/batchnorm/mul?
&layer_normalization_26/batchnorm/mul_1Muladd:z:0(layer_normalization_26/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????d@2(
&layer_normalization_26/batchnorm/mul_1?
&layer_normalization_26/batchnorm/mul_2Mul,layer_normalization_26/moments/mean:output:0(layer_normalization_26/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????d@2(
&layer_normalization_26/batchnorm/mul_2?
/layer_normalization_26/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_26_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype021
/layer_normalization_26/batchnorm/ReadVariableOp?
$layer_normalization_26/batchnorm/subSub7layer_normalization_26/batchnorm/ReadVariableOp:value:0*layer_normalization_26/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????d@2&
$layer_normalization_26/batchnorm/sub?
&layer_normalization_26/batchnorm/add_1AddV2*layer_normalization_26/batchnorm/mul_1:z:0(layer_normalization_26/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????d@2(
&layer_normalization_26/batchnorm/add_1?
/sequential_8/dense_100/Tensordot/ReadVariableOpReadVariableOp8sequential_8_dense_100_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype021
/sequential_8/dense_100/Tensordot/ReadVariableOp?
%sequential_8/dense_100/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_8/dense_100/Tensordot/axes?
%sequential_8/dense_100/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_8/dense_100/Tensordot/free?
&sequential_8/dense_100/Tensordot/ShapeShape*layer_normalization_26/batchnorm/add_1:z:0*
T0*
_output_shapes
:2(
&sequential_8/dense_100/Tensordot/Shape?
.sequential_8/dense_100/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_8/dense_100/Tensordot/GatherV2/axis?
)sequential_8/dense_100/Tensordot/GatherV2GatherV2/sequential_8/dense_100/Tensordot/Shape:output:0.sequential_8/dense_100/Tensordot/free:output:07sequential_8/dense_100/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_8/dense_100/Tensordot/GatherV2?
0sequential_8/dense_100/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_8/dense_100/Tensordot/GatherV2_1/axis?
+sequential_8/dense_100/Tensordot/GatherV2_1GatherV2/sequential_8/dense_100/Tensordot/Shape:output:0.sequential_8/dense_100/Tensordot/axes:output:09sequential_8/dense_100/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_8/dense_100/Tensordot/GatherV2_1?
&sequential_8/dense_100/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_8/dense_100/Tensordot/Const?
%sequential_8/dense_100/Tensordot/ProdProd2sequential_8/dense_100/Tensordot/GatherV2:output:0/sequential_8/dense_100/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_8/dense_100/Tensordot/Prod?
(sequential_8/dense_100/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_8/dense_100/Tensordot/Const_1?
'sequential_8/dense_100/Tensordot/Prod_1Prod4sequential_8/dense_100/Tensordot/GatherV2_1:output:01sequential_8/dense_100/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_8/dense_100/Tensordot/Prod_1?
,sequential_8/dense_100/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_8/dense_100/Tensordot/concat/axis?
'sequential_8/dense_100/Tensordot/concatConcatV2.sequential_8/dense_100/Tensordot/free:output:0.sequential_8/dense_100/Tensordot/axes:output:05sequential_8/dense_100/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_8/dense_100/Tensordot/concat?
&sequential_8/dense_100/Tensordot/stackPack.sequential_8/dense_100/Tensordot/Prod:output:00sequential_8/dense_100/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_8/dense_100/Tensordot/stack?
*sequential_8/dense_100/Tensordot/transpose	Transpose*layer_normalization_26/batchnorm/add_1:z:00sequential_8/dense_100/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????d@2,
*sequential_8/dense_100/Tensordot/transpose?
(sequential_8/dense_100/Tensordot/ReshapeReshape.sequential_8/dense_100/Tensordot/transpose:y:0/sequential_8/dense_100/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(sequential_8/dense_100/Tensordot/Reshape?
'sequential_8/dense_100/Tensordot/MatMulMatMul1sequential_8/dense_100/Tensordot/Reshape:output:07sequential_8/dense_100/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2)
'sequential_8/dense_100/Tensordot/MatMul?
(sequential_8/dense_100/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_8/dense_100/Tensordot/Const_2?
.sequential_8/dense_100/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_8/dense_100/Tensordot/concat_1/axis?
)sequential_8/dense_100/Tensordot/concat_1ConcatV22sequential_8/dense_100/Tensordot/GatherV2:output:01sequential_8/dense_100/Tensordot/Const_2:output:07sequential_8/dense_100/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_8/dense_100/Tensordot/concat_1?
 sequential_8/dense_100/TensordotReshape1sequential_8/dense_100/Tensordot/MatMul:product:02sequential_8/dense_100/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????d 2"
 sequential_8/dense_100/Tensordot?
-sequential_8/dense_100/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_dense_100_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_8/dense_100/BiasAdd/ReadVariableOp?
sequential_8/dense_100/BiasAddBiasAdd)sequential_8/dense_100/Tensordot:output:05sequential_8/dense_100/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d 2 
sequential_8/dense_100/BiasAdd?
sequential_8/dense_100/ReluRelu'sequential_8/dense_100/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d 2
sequential_8/dense_100/Relu?
/sequential_8/dense_101/Tensordot/ReadVariableOpReadVariableOp8sequential_8_dense_101_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype021
/sequential_8/dense_101/Tensordot/ReadVariableOp?
%sequential_8/dense_101/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_8/dense_101/Tensordot/axes?
%sequential_8/dense_101/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_8/dense_101/Tensordot/free?
&sequential_8/dense_101/Tensordot/ShapeShape)sequential_8/dense_100/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_8/dense_101/Tensordot/Shape?
.sequential_8/dense_101/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_8/dense_101/Tensordot/GatherV2/axis?
)sequential_8/dense_101/Tensordot/GatherV2GatherV2/sequential_8/dense_101/Tensordot/Shape:output:0.sequential_8/dense_101/Tensordot/free:output:07sequential_8/dense_101/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_8/dense_101/Tensordot/GatherV2?
0sequential_8/dense_101/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_8/dense_101/Tensordot/GatherV2_1/axis?
+sequential_8/dense_101/Tensordot/GatherV2_1GatherV2/sequential_8/dense_101/Tensordot/Shape:output:0.sequential_8/dense_101/Tensordot/axes:output:09sequential_8/dense_101/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_8/dense_101/Tensordot/GatherV2_1?
&sequential_8/dense_101/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_8/dense_101/Tensordot/Const?
%sequential_8/dense_101/Tensordot/ProdProd2sequential_8/dense_101/Tensordot/GatherV2:output:0/sequential_8/dense_101/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_8/dense_101/Tensordot/Prod?
(sequential_8/dense_101/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_8/dense_101/Tensordot/Const_1?
'sequential_8/dense_101/Tensordot/Prod_1Prod4sequential_8/dense_101/Tensordot/GatherV2_1:output:01sequential_8/dense_101/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_8/dense_101/Tensordot/Prod_1?
,sequential_8/dense_101/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_8/dense_101/Tensordot/concat/axis?
'sequential_8/dense_101/Tensordot/concatConcatV2.sequential_8/dense_101/Tensordot/free:output:0.sequential_8/dense_101/Tensordot/axes:output:05sequential_8/dense_101/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_8/dense_101/Tensordot/concat?
&sequential_8/dense_101/Tensordot/stackPack.sequential_8/dense_101/Tensordot/Prod:output:00sequential_8/dense_101/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_8/dense_101/Tensordot/stack?
*sequential_8/dense_101/Tensordot/transpose	Transpose)sequential_8/dense_100/Relu:activations:00sequential_8/dense_101/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????d 2,
*sequential_8/dense_101/Tensordot/transpose?
(sequential_8/dense_101/Tensordot/ReshapeReshape.sequential_8/dense_101/Tensordot/transpose:y:0/sequential_8/dense_101/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(sequential_8/dense_101/Tensordot/Reshape?
'sequential_8/dense_101/Tensordot/MatMulMatMul1sequential_8/dense_101/Tensordot/Reshape:output:07sequential_8/dense_101/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2)
'sequential_8/dense_101/Tensordot/MatMul?
(sequential_8/dense_101/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2*
(sequential_8/dense_101/Tensordot/Const_2?
.sequential_8/dense_101/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_8/dense_101/Tensordot/concat_1/axis?
)sequential_8/dense_101/Tensordot/concat_1ConcatV22sequential_8/dense_101/Tensordot/GatherV2:output:01sequential_8/dense_101/Tensordot/Const_2:output:07sequential_8/dense_101/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_8/dense_101/Tensordot/concat_1?
 sequential_8/dense_101/TensordotReshape1sequential_8/dense_101/Tensordot/MatMul:product:02sequential_8/dense_101/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????d@2"
 sequential_8/dense_101/Tensordot?
-sequential_8/dense_101/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_dense_101_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-sequential_8/dense_101/BiasAdd/ReadVariableOp?
sequential_8/dense_101/BiasAddBiasAdd)sequential_8/dense_101/Tensordot:output:05sequential_8/dense_101/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@2 
sequential_8/dense_101/BiasAddy
dropout_43/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_43/dropout/Const?
dropout_43/dropout/MulMul'sequential_8/dense_101/BiasAdd:output:0!dropout_43/dropout/Const:output:0*
T0*+
_output_shapes
:?????????d@2
dropout_43/dropout/Mul?
dropout_43/dropout/ShapeShape'sequential_8/dense_101/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_43/dropout/Shape?
/dropout_43/dropout/random_uniform/RandomUniformRandomUniform!dropout_43/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????d@*
dtype021
/dropout_43/dropout/random_uniform/RandomUniform?
!dropout_43/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2#
!dropout_43/dropout/GreaterEqual/y?
dropout_43/dropout/GreaterEqualGreaterEqual8dropout_43/dropout/random_uniform/RandomUniform:output:0*dropout_43/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????d@2!
dropout_43/dropout/GreaterEqual?
dropout_43/dropout/CastCast#dropout_43/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????d@2
dropout_43/dropout/Cast?
dropout_43/dropout/Mul_1Muldropout_43/dropout/Mul:z:0dropout_43/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????d@2
dropout_43/dropout/Mul_1?
add_1AddV2*layer_normalization_26/batchnorm/add_1:z:0dropout_43/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????d@2
add_1?
5layer_normalization_27/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_27/moments/mean/reduction_indices?
#layer_normalization_27/moments/meanMean	add_1:z:0>layer_normalization_27/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????d*
	keep_dims(2%
#layer_normalization_27/moments/mean?
+layer_normalization_27/moments/StopGradientStopGradient,layer_normalization_27/moments/mean:output:0*
T0*+
_output_shapes
:?????????d2-
+layer_normalization_27/moments/StopGradient?
0layer_normalization_27/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_27/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????d@22
0layer_normalization_27/moments/SquaredDifference?
9layer_normalization_27/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_27/moments/variance/reduction_indices?
'layer_normalization_27/moments/varianceMean4layer_normalization_27/moments/SquaredDifference:z:0Blayer_normalization_27/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????d*
	keep_dims(2)
'layer_normalization_27/moments/variance?
&layer_normalization_27/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52(
&layer_normalization_27/batchnorm/add/y?
$layer_normalization_27/batchnorm/addAddV20layer_normalization_27/moments/variance:output:0/layer_normalization_27/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????d2&
$layer_normalization_27/batchnorm/add?
&layer_normalization_27/batchnorm/RsqrtRsqrt(layer_normalization_27/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????d2(
&layer_normalization_27/batchnorm/Rsqrt?
3layer_normalization_27/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_27_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype025
3layer_normalization_27/batchnorm/mul/ReadVariableOp?
$layer_normalization_27/batchnorm/mulMul*layer_normalization_27/batchnorm/Rsqrt:y:0;layer_normalization_27/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@2&
$layer_normalization_27/batchnorm/mul?
&layer_normalization_27/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_27/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????d@2(
&layer_normalization_27/batchnorm/mul_1?
&layer_normalization_27/batchnorm/mul_2Mul,layer_normalization_27/moments/mean:output:0(layer_normalization_27/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????d@2(
&layer_normalization_27/batchnorm/mul_2?
/layer_normalization_27/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_27_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype021
/layer_normalization_27/batchnorm/ReadVariableOp?
$layer_normalization_27/batchnorm/subSub7layer_normalization_27/batchnorm/ReadVariableOp:value:0*layer_normalization_27/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????d@2&
$layer_normalization_27/batchnorm/sub?
&layer_normalization_27/batchnorm/add_1AddV2*layer_normalization_27/batchnorm/mul_1:z:0(layer_normalization_27/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????d@2(
&layer_normalization_27/batchnorm/add_1?
IdentityIdentity*layer_normalization_27/batchnorm/add_1:z:00^layer_normalization_26/batchnorm/ReadVariableOp4^layer_normalization_26/batchnorm/mul/ReadVariableOp0^layer_normalization_27/batchnorm/ReadVariableOp4^layer_normalization_27/batchnorm/mul/ReadVariableOp7^multi_head_attention_8/dense_96/BiasAdd/ReadVariableOp9^multi_head_attention_8/dense_96/Tensordot/ReadVariableOp7^multi_head_attention_8/dense_97/BiasAdd/ReadVariableOp9^multi_head_attention_8/dense_97/Tensordot/ReadVariableOp7^multi_head_attention_8/dense_98/BiasAdd/ReadVariableOp9^multi_head_attention_8/dense_98/Tensordot/ReadVariableOp7^multi_head_attention_8/dense_99/BiasAdd/ReadVariableOp9^multi_head_attention_8/dense_99/Tensordot/ReadVariableOp.^sequential_8/dense_100/BiasAdd/ReadVariableOp0^sequential_8/dense_100/Tensordot/ReadVariableOp.^sequential_8/dense_101/BiasAdd/ReadVariableOp0^sequential_8/dense_101/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????d@: : : : : : : : : : : : : : : : 2b
/layer_normalization_26/batchnorm/ReadVariableOp/layer_normalization_26/batchnorm/ReadVariableOp2j
3layer_normalization_26/batchnorm/mul/ReadVariableOp3layer_normalization_26/batchnorm/mul/ReadVariableOp2b
/layer_normalization_27/batchnorm/ReadVariableOp/layer_normalization_27/batchnorm/ReadVariableOp2j
3layer_normalization_27/batchnorm/mul/ReadVariableOp3layer_normalization_27/batchnorm/mul/ReadVariableOp2p
6multi_head_attention_8/dense_96/BiasAdd/ReadVariableOp6multi_head_attention_8/dense_96/BiasAdd/ReadVariableOp2t
8multi_head_attention_8/dense_96/Tensordot/ReadVariableOp8multi_head_attention_8/dense_96/Tensordot/ReadVariableOp2p
6multi_head_attention_8/dense_97/BiasAdd/ReadVariableOp6multi_head_attention_8/dense_97/BiasAdd/ReadVariableOp2t
8multi_head_attention_8/dense_97/Tensordot/ReadVariableOp8multi_head_attention_8/dense_97/Tensordot/ReadVariableOp2p
6multi_head_attention_8/dense_98/BiasAdd/ReadVariableOp6multi_head_attention_8/dense_98/BiasAdd/ReadVariableOp2t
8multi_head_attention_8/dense_98/Tensordot/ReadVariableOp8multi_head_attention_8/dense_98/Tensordot/ReadVariableOp2p
6multi_head_attention_8/dense_99/BiasAdd/ReadVariableOp6multi_head_attention_8/dense_99/BiasAdd/ReadVariableOp2t
8multi_head_attention_8/dense_99/Tensordot/ReadVariableOp8multi_head_attention_8/dense_99/Tensordot/ReadVariableOp2^
-sequential_8/dense_100/BiasAdd/ReadVariableOp-sequential_8/dense_100/BiasAdd/ReadVariableOp2b
/sequential_8/dense_100/Tensordot/ReadVariableOp/sequential_8/dense_100/Tensordot/ReadVariableOp2^
-sequential_8/dense_101/BiasAdd/ReadVariableOp-sequential_8/dense_101/BiasAdd/ReadVariableOp2b
/sequential_8/dense_101/Tensordot/ReadVariableOp/sequential_8/dense_101/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????d@
 
_user_specified_nameinputs
?
?
G__inference_sequential_8_layer_call_and_return_conditional_losses_90952

inputs!
dense_100_90941:@ 
dense_100_90943: !
dense_101_90946: @
dense_101_90948:@
identity??!dense_100/StatefulPartitionedCall?!dense_101/StatefulPartitionedCall?
!dense_100/StatefulPartitionedCallStatefulPartitionedCallinputsdense_100_90941dense_100_90943*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_100_layer_call_and_return_conditional_losses_908492#
!dense_100/StatefulPartitionedCall?
!dense_101/StatefulPartitionedCallStatefulPartitionedCall*dense_100/StatefulPartitionedCall:output:0dense_101_90946dense_101_90948*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_101_layer_call_and_return_conditional_losses_908852#
!dense_101/StatefulPartitionedCall?
IdentityIdentity*dense_101/StatefulPartitionedCall:output:0"^dense_100/StatefulPartitionedCall"^dense_101/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????d@: : : : 2F
!dense_100/StatefulPartitionedCall!dense_100/StatefulPartitionedCall2F
!dense_101/StatefulPartitionedCall!dense_101/StatefulPartitionedCall:S O
+
_output_shapes
:?????????d@
 
_user_specified_nameinputs
?
d
E__inference_dropout_44_layer_call_and_return_conditional_losses_91509

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
??
?
N__inference_transformer_block_8_layer_call_and_return_conditional_losses_93528

inputsS
Amulti_head_attention_8_dense_96_tensordot_readvariableop_resource:@@M
?multi_head_attention_8_dense_96_biasadd_readvariableop_resource:@S
Amulti_head_attention_8_dense_97_tensordot_readvariableop_resource:@@M
?multi_head_attention_8_dense_97_biasadd_readvariableop_resource:@S
Amulti_head_attention_8_dense_98_tensordot_readvariableop_resource:@@M
?multi_head_attention_8_dense_98_biasadd_readvariableop_resource:@S
Amulti_head_attention_8_dense_99_tensordot_readvariableop_resource:@@M
?multi_head_attention_8_dense_99_biasadd_readvariableop_resource:@J
<layer_normalization_26_batchnorm_mul_readvariableop_resource:@F
8layer_normalization_26_batchnorm_readvariableop_resource:@J
8sequential_8_dense_100_tensordot_readvariableop_resource:@ D
6sequential_8_dense_100_biasadd_readvariableop_resource: J
8sequential_8_dense_101_tensordot_readvariableop_resource: @D
6sequential_8_dense_101_biasadd_readvariableop_resource:@J
<layer_normalization_27_batchnorm_mul_readvariableop_resource:@F
8layer_normalization_27_batchnorm_readvariableop_resource:@
identity??/layer_normalization_26/batchnorm/ReadVariableOp?3layer_normalization_26/batchnorm/mul/ReadVariableOp?/layer_normalization_27/batchnorm/ReadVariableOp?3layer_normalization_27/batchnorm/mul/ReadVariableOp?6multi_head_attention_8/dense_96/BiasAdd/ReadVariableOp?8multi_head_attention_8/dense_96/Tensordot/ReadVariableOp?6multi_head_attention_8/dense_97/BiasAdd/ReadVariableOp?8multi_head_attention_8/dense_97/Tensordot/ReadVariableOp?6multi_head_attention_8/dense_98/BiasAdd/ReadVariableOp?8multi_head_attention_8/dense_98/Tensordot/ReadVariableOp?6multi_head_attention_8/dense_99/BiasAdd/ReadVariableOp?8multi_head_attention_8/dense_99/Tensordot/ReadVariableOp?-sequential_8/dense_100/BiasAdd/ReadVariableOp?/sequential_8/dense_100/Tensordot/ReadVariableOp?-sequential_8/dense_101/BiasAdd/ReadVariableOp?/sequential_8/dense_101/Tensordot/ReadVariableOpr
multi_head_attention_8/ShapeShapeinputs*
T0*
_output_shapes
:2
multi_head_attention_8/Shape?
*multi_head_attention_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*multi_head_attention_8/strided_slice/stack?
,multi_head_attention_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,multi_head_attention_8/strided_slice/stack_1?
,multi_head_attention_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,multi_head_attention_8/strided_slice/stack_2?
$multi_head_attention_8/strided_sliceStridedSlice%multi_head_attention_8/Shape:output:03multi_head_attention_8/strided_slice/stack:output:05multi_head_attention_8/strided_slice/stack_1:output:05multi_head_attention_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$multi_head_attention_8/strided_slice?
8multi_head_attention_8/dense_96/Tensordot/ReadVariableOpReadVariableOpAmulti_head_attention_8_dense_96_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02:
8multi_head_attention_8/dense_96/Tensordot/ReadVariableOp?
.multi_head_attention_8/dense_96/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:20
.multi_head_attention_8/dense_96/Tensordot/axes?
.multi_head_attention_8/dense_96/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       20
.multi_head_attention_8/dense_96/Tensordot/free?
/multi_head_attention_8/dense_96/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:21
/multi_head_attention_8/dense_96/Tensordot/Shape?
7multi_head_attention_8/dense_96/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_attention_8/dense_96/Tensordot/GatherV2/axis?
2multi_head_attention_8/dense_96/Tensordot/GatherV2GatherV28multi_head_attention_8/dense_96/Tensordot/Shape:output:07multi_head_attention_8/dense_96/Tensordot/free:output:0@multi_head_attention_8/dense_96/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:24
2multi_head_attention_8/dense_96/Tensordot/GatherV2?
9multi_head_attention_8/dense_96/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9multi_head_attention_8/dense_96/Tensordot/GatherV2_1/axis?
4multi_head_attention_8/dense_96/Tensordot/GatherV2_1GatherV28multi_head_attention_8/dense_96/Tensordot/Shape:output:07multi_head_attention_8/dense_96/Tensordot/axes:output:0Bmulti_head_attention_8/dense_96/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4multi_head_attention_8/dense_96/Tensordot/GatherV2_1?
/multi_head_attention_8/dense_96/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/multi_head_attention_8/dense_96/Tensordot/Const?
.multi_head_attention_8/dense_96/Tensordot/ProdProd;multi_head_attention_8/dense_96/Tensordot/GatherV2:output:08multi_head_attention_8/dense_96/Tensordot/Const:output:0*
T0*
_output_shapes
: 20
.multi_head_attention_8/dense_96/Tensordot/Prod?
1multi_head_attention_8/dense_96/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1multi_head_attention_8/dense_96/Tensordot/Const_1?
0multi_head_attention_8/dense_96/Tensordot/Prod_1Prod=multi_head_attention_8/dense_96/Tensordot/GatherV2_1:output:0:multi_head_attention_8/dense_96/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 22
0multi_head_attention_8/dense_96/Tensordot/Prod_1?
5multi_head_attention_8/dense_96/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5multi_head_attention_8/dense_96/Tensordot/concat/axis?
0multi_head_attention_8/dense_96/Tensordot/concatConcatV27multi_head_attention_8/dense_96/Tensordot/free:output:07multi_head_attention_8/dense_96/Tensordot/axes:output:0>multi_head_attention_8/dense_96/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:22
0multi_head_attention_8/dense_96/Tensordot/concat?
/multi_head_attention_8/dense_96/Tensordot/stackPack7multi_head_attention_8/dense_96/Tensordot/Prod:output:09multi_head_attention_8/dense_96/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:21
/multi_head_attention_8/dense_96/Tensordot/stack?
3multi_head_attention_8/dense_96/Tensordot/transpose	Transposeinputs9multi_head_attention_8/dense_96/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????d@25
3multi_head_attention_8/dense_96/Tensordot/transpose?
1multi_head_attention_8/dense_96/Tensordot/ReshapeReshape7multi_head_attention_8/dense_96/Tensordot/transpose:y:08multi_head_attention_8/dense_96/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????23
1multi_head_attention_8/dense_96/Tensordot/Reshape?
0multi_head_attention_8/dense_96/Tensordot/MatMulMatMul:multi_head_attention_8/dense_96/Tensordot/Reshape:output:0@multi_head_attention_8/dense_96/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@22
0multi_head_attention_8/dense_96/Tensordot/MatMul?
1multi_head_attention_8/dense_96/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@23
1multi_head_attention_8/dense_96/Tensordot/Const_2?
7multi_head_attention_8/dense_96/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_attention_8/dense_96/Tensordot/concat_1/axis?
2multi_head_attention_8/dense_96/Tensordot/concat_1ConcatV2;multi_head_attention_8/dense_96/Tensordot/GatherV2:output:0:multi_head_attention_8/dense_96/Tensordot/Const_2:output:0@multi_head_attention_8/dense_96/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:24
2multi_head_attention_8/dense_96/Tensordot/concat_1?
)multi_head_attention_8/dense_96/TensordotReshape:multi_head_attention_8/dense_96/Tensordot/MatMul:product:0;multi_head_attention_8/dense_96/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????d@2+
)multi_head_attention_8/dense_96/Tensordot?
6multi_head_attention_8/dense_96/BiasAdd/ReadVariableOpReadVariableOp?multi_head_attention_8_dense_96_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6multi_head_attention_8/dense_96/BiasAdd/ReadVariableOp?
'multi_head_attention_8/dense_96/BiasAddBiasAdd2multi_head_attention_8/dense_96/Tensordot:output:0>multi_head_attention_8/dense_96/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@2)
'multi_head_attention_8/dense_96/BiasAdd?
8multi_head_attention_8/dense_97/Tensordot/ReadVariableOpReadVariableOpAmulti_head_attention_8_dense_97_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02:
8multi_head_attention_8/dense_97/Tensordot/ReadVariableOp?
.multi_head_attention_8/dense_97/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:20
.multi_head_attention_8/dense_97/Tensordot/axes?
.multi_head_attention_8/dense_97/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       20
.multi_head_attention_8/dense_97/Tensordot/free?
/multi_head_attention_8/dense_97/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:21
/multi_head_attention_8/dense_97/Tensordot/Shape?
7multi_head_attention_8/dense_97/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_attention_8/dense_97/Tensordot/GatherV2/axis?
2multi_head_attention_8/dense_97/Tensordot/GatherV2GatherV28multi_head_attention_8/dense_97/Tensordot/Shape:output:07multi_head_attention_8/dense_97/Tensordot/free:output:0@multi_head_attention_8/dense_97/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:24
2multi_head_attention_8/dense_97/Tensordot/GatherV2?
9multi_head_attention_8/dense_97/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9multi_head_attention_8/dense_97/Tensordot/GatherV2_1/axis?
4multi_head_attention_8/dense_97/Tensordot/GatherV2_1GatherV28multi_head_attention_8/dense_97/Tensordot/Shape:output:07multi_head_attention_8/dense_97/Tensordot/axes:output:0Bmulti_head_attention_8/dense_97/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4multi_head_attention_8/dense_97/Tensordot/GatherV2_1?
/multi_head_attention_8/dense_97/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/multi_head_attention_8/dense_97/Tensordot/Const?
.multi_head_attention_8/dense_97/Tensordot/ProdProd;multi_head_attention_8/dense_97/Tensordot/GatherV2:output:08multi_head_attention_8/dense_97/Tensordot/Const:output:0*
T0*
_output_shapes
: 20
.multi_head_attention_8/dense_97/Tensordot/Prod?
1multi_head_attention_8/dense_97/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1multi_head_attention_8/dense_97/Tensordot/Const_1?
0multi_head_attention_8/dense_97/Tensordot/Prod_1Prod=multi_head_attention_8/dense_97/Tensordot/GatherV2_1:output:0:multi_head_attention_8/dense_97/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 22
0multi_head_attention_8/dense_97/Tensordot/Prod_1?
5multi_head_attention_8/dense_97/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5multi_head_attention_8/dense_97/Tensordot/concat/axis?
0multi_head_attention_8/dense_97/Tensordot/concatConcatV27multi_head_attention_8/dense_97/Tensordot/free:output:07multi_head_attention_8/dense_97/Tensordot/axes:output:0>multi_head_attention_8/dense_97/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:22
0multi_head_attention_8/dense_97/Tensordot/concat?
/multi_head_attention_8/dense_97/Tensordot/stackPack7multi_head_attention_8/dense_97/Tensordot/Prod:output:09multi_head_attention_8/dense_97/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:21
/multi_head_attention_8/dense_97/Tensordot/stack?
3multi_head_attention_8/dense_97/Tensordot/transpose	Transposeinputs9multi_head_attention_8/dense_97/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????d@25
3multi_head_attention_8/dense_97/Tensordot/transpose?
1multi_head_attention_8/dense_97/Tensordot/ReshapeReshape7multi_head_attention_8/dense_97/Tensordot/transpose:y:08multi_head_attention_8/dense_97/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????23
1multi_head_attention_8/dense_97/Tensordot/Reshape?
0multi_head_attention_8/dense_97/Tensordot/MatMulMatMul:multi_head_attention_8/dense_97/Tensordot/Reshape:output:0@multi_head_attention_8/dense_97/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@22
0multi_head_attention_8/dense_97/Tensordot/MatMul?
1multi_head_attention_8/dense_97/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@23
1multi_head_attention_8/dense_97/Tensordot/Const_2?
7multi_head_attention_8/dense_97/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_attention_8/dense_97/Tensordot/concat_1/axis?
2multi_head_attention_8/dense_97/Tensordot/concat_1ConcatV2;multi_head_attention_8/dense_97/Tensordot/GatherV2:output:0:multi_head_attention_8/dense_97/Tensordot/Const_2:output:0@multi_head_attention_8/dense_97/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:24
2multi_head_attention_8/dense_97/Tensordot/concat_1?
)multi_head_attention_8/dense_97/TensordotReshape:multi_head_attention_8/dense_97/Tensordot/MatMul:product:0;multi_head_attention_8/dense_97/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????d@2+
)multi_head_attention_8/dense_97/Tensordot?
6multi_head_attention_8/dense_97/BiasAdd/ReadVariableOpReadVariableOp?multi_head_attention_8_dense_97_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6multi_head_attention_8/dense_97/BiasAdd/ReadVariableOp?
'multi_head_attention_8/dense_97/BiasAddBiasAdd2multi_head_attention_8/dense_97/Tensordot:output:0>multi_head_attention_8/dense_97/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@2)
'multi_head_attention_8/dense_97/BiasAdd?
8multi_head_attention_8/dense_98/Tensordot/ReadVariableOpReadVariableOpAmulti_head_attention_8_dense_98_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02:
8multi_head_attention_8/dense_98/Tensordot/ReadVariableOp?
.multi_head_attention_8/dense_98/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:20
.multi_head_attention_8/dense_98/Tensordot/axes?
.multi_head_attention_8/dense_98/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       20
.multi_head_attention_8/dense_98/Tensordot/free?
/multi_head_attention_8/dense_98/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:21
/multi_head_attention_8/dense_98/Tensordot/Shape?
7multi_head_attention_8/dense_98/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_attention_8/dense_98/Tensordot/GatherV2/axis?
2multi_head_attention_8/dense_98/Tensordot/GatherV2GatherV28multi_head_attention_8/dense_98/Tensordot/Shape:output:07multi_head_attention_8/dense_98/Tensordot/free:output:0@multi_head_attention_8/dense_98/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:24
2multi_head_attention_8/dense_98/Tensordot/GatherV2?
9multi_head_attention_8/dense_98/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9multi_head_attention_8/dense_98/Tensordot/GatherV2_1/axis?
4multi_head_attention_8/dense_98/Tensordot/GatherV2_1GatherV28multi_head_attention_8/dense_98/Tensordot/Shape:output:07multi_head_attention_8/dense_98/Tensordot/axes:output:0Bmulti_head_attention_8/dense_98/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4multi_head_attention_8/dense_98/Tensordot/GatherV2_1?
/multi_head_attention_8/dense_98/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/multi_head_attention_8/dense_98/Tensordot/Const?
.multi_head_attention_8/dense_98/Tensordot/ProdProd;multi_head_attention_8/dense_98/Tensordot/GatherV2:output:08multi_head_attention_8/dense_98/Tensordot/Const:output:0*
T0*
_output_shapes
: 20
.multi_head_attention_8/dense_98/Tensordot/Prod?
1multi_head_attention_8/dense_98/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1multi_head_attention_8/dense_98/Tensordot/Const_1?
0multi_head_attention_8/dense_98/Tensordot/Prod_1Prod=multi_head_attention_8/dense_98/Tensordot/GatherV2_1:output:0:multi_head_attention_8/dense_98/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 22
0multi_head_attention_8/dense_98/Tensordot/Prod_1?
5multi_head_attention_8/dense_98/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5multi_head_attention_8/dense_98/Tensordot/concat/axis?
0multi_head_attention_8/dense_98/Tensordot/concatConcatV27multi_head_attention_8/dense_98/Tensordot/free:output:07multi_head_attention_8/dense_98/Tensordot/axes:output:0>multi_head_attention_8/dense_98/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:22
0multi_head_attention_8/dense_98/Tensordot/concat?
/multi_head_attention_8/dense_98/Tensordot/stackPack7multi_head_attention_8/dense_98/Tensordot/Prod:output:09multi_head_attention_8/dense_98/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:21
/multi_head_attention_8/dense_98/Tensordot/stack?
3multi_head_attention_8/dense_98/Tensordot/transpose	Transposeinputs9multi_head_attention_8/dense_98/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????d@25
3multi_head_attention_8/dense_98/Tensordot/transpose?
1multi_head_attention_8/dense_98/Tensordot/ReshapeReshape7multi_head_attention_8/dense_98/Tensordot/transpose:y:08multi_head_attention_8/dense_98/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????23
1multi_head_attention_8/dense_98/Tensordot/Reshape?
0multi_head_attention_8/dense_98/Tensordot/MatMulMatMul:multi_head_attention_8/dense_98/Tensordot/Reshape:output:0@multi_head_attention_8/dense_98/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@22
0multi_head_attention_8/dense_98/Tensordot/MatMul?
1multi_head_attention_8/dense_98/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@23
1multi_head_attention_8/dense_98/Tensordot/Const_2?
7multi_head_attention_8/dense_98/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_attention_8/dense_98/Tensordot/concat_1/axis?
2multi_head_attention_8/dense_98/Tensordot/concat_1ConcatV2;multi_head_attention_8/dense_98/Tensordot/GatherV2:output:0:multi_head_attention_8/dense_98/Tensordot/Const_2:output:0@multi_head_attention_8/dense_98/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:24
2multi_head_attention_8/dense_98/Tensordot/concat_1?
)multi_head_attention_8/dense_98/TensordotReshape:multi_head_attention_8/dense_98/Tensordot/MatMul:product:0;multi_head_attention_8/dense_98/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????d@2+
)multi_head_attention_8/dense_98/Tensordot?
6multi_head_attention_8/dense_98/BiasAdd/ReadVariableOpReadVariableOp?multi_head_attention_8_dense_98_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6multi_head_attention_8/dense_98/BiasAdd/ReadVariableOp?
'multi_head_attention_8/dense_98/BiasAddBiasAdd2multi_head_attention_8/dense_98/Tensordot:output:0>multi_head_attention_8/dense_98/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@2)
'multi_head_attention_8/dense_98/BiasAdd?
&multi_head_attention_8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&multi_head_attention_8/Reshape/shape/1?
&multi_head_attention_8/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2(
&multi_head_attention_8/Reshape/shape/2?
&multi_head_attention_8/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2(
&multi_head_attention_8/Reshape/shape/3?
$multi_head_attention_8/Reshape/shapePack-multi_head_attention_8/strided_slice:output:0/multi_head_attention_8/Reshape/shape/1:output:0/multi_head_attention_8/Reshape/shape/2:output:0/multi_head_attention_8/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2&
$multi_head_attention_8/Reshape/shape?
multi_head_attention_8/ReshapeReshape0multi_head_attention_8/dense_96/BiasAdd:output:0-multi_head_attention_8/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2 
multi_head_attention_8/Reshape?
%multi_head_attention_8/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2'
%multi_head_attention_8/transpose/perm?
 multi_head_attention_8/transpose	Transpose'multi_head_attention_8/Reshape:output:0.multi_head_attention_8/transpose/perm:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2"
 multi_head_attention_8/transpose?
(multi_head_attention_8/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(multi_head_attention_8/Reshape_1/shape/1?
(multi_head_attention_8/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(multi_head_attention_8/Reshape_1/shape/2?
(multi_head_attention_8/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2*
(multi_head_attention_8/Reshape_1/shape/3?
&multi_head_attention_8/Reshape_1/shapePack-multi_head_attention_8/strided_slice:output:01multi_head_attention_8/Reshape_1/shape/1:output:01multi_head_attention_8/Reshape_1/shape/2:output:01multi_head_attention_8/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_attention_8/Reshape_1/shape?
 multi_head_attention_8/Reshape_1Reshape0multi_head_attention_8/dense_97/BiasAdd:output:0/multi_head_attention_8/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2"
 multi_head_attention_8/Reshape_1?
'multi_head_attention_8/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2)
'multi_head_attention_8/transpose_1/perm?
"multi_head_attention_8/transpose_1	Transpose)multi_head_attention_8/Reshape_1:output:00multi_head_attention_8/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2$
"multi_head_attention_8/transpose_1?
(multi_head_attention_8/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(multi_head_attention_8/Reshape_2/shape/1?
(multi_head_attention_8/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(multi_head_attention_8/Reshape_2/shape/2?
(multi_head_attention_8/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2*
(multi_head_attention_8/Reshape_2/shape/3?
&multi_head_attention_8/Reshape_2/shapePack-multi_head_attention_8/strided_slice:output:01multi_head_attention_8/Reshape_2/shape/1:output:01multi_head_attention_8/Reshape_2/shape/2:output:01multi_head_attention_8/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_attention_8/Reshape_2/shape?
 multi_head_attention_8/Reshape_2Reshape0multi_head_attention_8/dense_98/BiasAdd:output:0/multi_head_attention_8/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2"
 multi_head_attention_8/Reshape_2?
'multi_head_attention_8/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2)
'multi_head_attention_8/transpose_2/perm?
"multi_head_attention_8/transpose_2	Transpose)multi_head_attention_8/Reshape_2:output:00multi_head_attention_8/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2$
"multi_head_attention_8/transpose_2?
multi_head_attention_8/MatMulBatchMatMulV2$multi_head_attention_8/transpose:y:0&multi_head_attention_8/transpose_1:y:0*
T0*A
_output_shapes/
-:+???????????????????????????*
adj_y(2
multi_head_attention_8/MatMul?
multi_head_attention_8/Shape_1Shape&multi_head_attention_8/transpose_1:y:0*
T0*
_output_shapes
:2 
multi_head_attention_8/Shape_1?
,multi_head_attention_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2.
,multi_head_attention_8/strided_slice_1/stack?
.multi_head_attention_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 20
.multi_head_attention_8/strided_slice_1/stack_1?
.multi_head_attention_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.multi_head_attention_8/strided_slice_1/stack_2?
&multi_head_attention_8/strided_slice_1StridedSlice'multi_head_attention_8/Shape_1:output:05multi_head_attention_8/strided_slice_1/stack:output:07multi_head_attention_8/strided_slice_1/stack_1:output:07multi_head_attention_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&multi_head_attention_8/strided_slice_1?
multi_head_attention_8/CastCast/multi_head_attention_8/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
multi_head_attention_8/Cast?
multi_head_attention_8/SqrtSqrtmulti_head_attention_8/Cast:y:0*
T0*
_output_shapes
: 2
multi_head_attention_8/Sqrt?
multi_head_attention_8/truedivRealDiv&multi_head_attention_8/MatMul:output:0multi_head_attention_8/Sqrt:y:0*
T0*A
_output_shapes/
-:+???????????????????????????2 
multi_head_attention_8/truediv?
multi_head_attention_8/SoftmaxSoftmax"multi_head_attention_8/truediv:z:0*
T0*A
_output_shapes/
-:+???????????????????????????2 
multi_head_attention_8/Softmax?
multi_head_attention_8/MatMul_1BatchMatMulV2(multi_head_attention_8/Softmax:softmax:0&multi_head_attention_8/transpose_2:y:0*
T0*8
_output_shapes&
$:"?????????????????? 2!
multi_head_attention_8/MatMul_1?
'multi_head_attention_8/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             2)
'multi_head_attention_8/transpose_3/perm?
"multi_head_attention_8/transpose_3	Transpose(multi_head_attention_8/MatMul_1:output:00multi_head_attention_8/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2$
"multi_head_attention_8/transpose_3?
(multi_head_attention_8/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(multi_head_attention_8/Reshape_3/shape/1?
(multi_head_attention_8/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@2*
(multi_head_attention_8/Reshape_3/shape/2?
&multi_head_attention_8/Reshape_3/shapePack-multi_head_attention_8/strided_slice:output:01multi_head_attention_8/Reshape_3/shape/1:output:01multi_head_attention_8/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_attention_8/Reshape_3/shape?
 multi_head_attention_8/Reshape_3Reshape&multi_head_attention_8/transpose_3:y:0/multi_head_attention_8/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :??????????????????@2"
 multi_head_attention_8/Reshape_3?
8multi_head_attention_8/dense_99/Tensordot/ReadVariableOpReadVariableOpAmulti_head_attention_8_dense_99_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02:
8multi_head_attention_8/dense_99/Tensordot/ReadVariableOp?
.multi_head_attention_8/dense_99/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:20
.multi_head_attention_8/dense_99/Tensordot/axes?
.multi_head_attention_8/dense_99/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       20
.multi_head_attention_8/dense_99/Tensordot/free?
/multi_head_attention_8/dense_99/Tensordot/ShapeShape)multi_head_attention_8/Reshape_3:output:0*
T0*
_output_shapes
:21
/multi_head_attention_8/dense_99/Tensordot/Shape?
7multi_head_attention_8/dense_99/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_attention_8/dense_99/Tensordot/GatherV2/axis?
2multi_head_attention_8/dense_99/Tensordot/GatherV2GatherV28multi_head_attention_8/dense_99/Tensordot/Shape:output:07multi_head_attention_8/dense_99/Tensordot/free:output:0@multi_head_attention_8/dense_99/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:24
2multi_head_attention_8/dense_99/Tensordot/GatherV2?
9multi_head_attention_8/dense_99/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9multi_head_attention_8/dense_99/Tensordot/GatherV2_1/axis?
4multi_head_attention_8/dense_99/Tensordot/GatherV2_1GatherV28multi_head_attention_8/dense_99/Tensordot/Shape:output:07multi_head_attention_8/dense_99/Tensordot/axes:output:0Bmulti_head_attention_8/dense_99/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4multi_head_attention_8/dense_99/Tensordot/GatherV2_1?
/multi_head_attention_8/dense_99/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/multi_head_attention_8/dense_99/Tensordot/Const?
.multi_head_attention_8/dense_99/Tensordot/ProdProd;multi_head_attention_8/dense_99/Tensordot/GatherV2:output:08multi_head_attention_8/dense_99/Tensordot/Const:output:0*
T0*
_output_shapes
: 20
.multi_head_attention_8/dense_99/Tensordot/Prod?
1multi_head_attention_8/dense_99/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1multi_head_attention_8/dense_99/Tensordot/Const_1?
0multi_head_attention_8/dense_99/Tensordot/Prod_1Prod=multi_head_attention_8/dense_99/Tensordot/GatherV2_1:output:0:multi_head_attention_8/dense_99/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 22
0multi_head_attention_8/dense_99/Tensordot/Prod_1?
5multi_head_attention_8/dense_99/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5multi_head_attention_8/dense_99/Tensordot/concat/axis?
0multi_head_attention_8/dense_99/Tensordot/concatConcatV27multi_head_attention_8/dense_99/Tensordot/free:output:07multi_head_attention_8/dense_99/Tensordot/axes:output:0>multi_head_attention_8/dense_99/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:22
0multi_head_attention_8/dense_99/Tensordot/concat?
/multi_head_attention_8/dense_99/Tensordot/stackPack7multi_head_attention_8/dense_99/Tensordot/Prod:output:09multi_head_attention_8/dense_99/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:21
/multi_head_attention_8/dense_99/Tensordot/stack?
3multi_head_attention_8/dense_99/Tensordot/transpose	Transpose)multi_head_attention_8/Reshape_3:output:09multi_head_attention_8/dense_99/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :??????????????????@25
3multi_head_attention_8/dense_99/Tensordot/transpose?
1multi_head_attention_8/dense_99/Tensordot/ReshapeReshape7multi_head_attention_8/dense_99/Tensordot/transpose:y:08multi_head_attention_8/dense_99/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????23
1multi_head_attention_8/dense_99/Tensordot/Reshape?
0multi_head_attention_8/dense_99/Tensordot/MatMulMatMul:multi_head_attention_8/dense_99/Tensordot/Reshape:output:0@multi_head_attention_8/dense_99/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@22
0multi_head_attention_8/dense_99/Tensordot/MatMul?
1multi_head_attention_8/dense_99/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@23
1multi_head_attention_8/dense_99/Tensordot/Const_2?
7multi_head_attention_8/dense_99/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_attention_8/dense_99/Tensordot/concat_1/axis?
2multi_head_attention_8/dense_99/Tensordot/concat_1ConcatV2;multi_head_attention_8/dense_99/Tensordot/GatherV2:output:0:multi_head_attention_8/dense_99/Tensordot/Const_2:output:0@multi_head_attention_8/dense_99/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:24
2multi_head_attention_8/dense_99/Tensordot/concat_1?
)multi_head_attention_8/dense_99/TensordotReshape:multi_head_attention_8/dense_99/Tensordot/MatMul:product:0;multi_head_attention_8/dense_99/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :??????????????????@2+
)multi_head_attention_8/dense_99/Tensordot?
6multi_head_attention_8/dense_99/BiasAdd/ReadVariableOpReadVariableOp?multi_head_attention_8_dense_99_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6multi_head_attention_8/dense_99/BiasAdd/ReadVariableOp?
'multi_head_attention_8/dense_99/BiasAddBiasAdd2multi_head_attention_8/dense_99/Tensordot:output:0>multi_head_attention_8/dense_99/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????@2)
'multi_head_attention_8/dense_99/BiasAddy
dropout_42/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_42/dropout/Const?
dropout_42/dropout/MulMul0multi_head_attention_8/dense_99/BiasAdd:output:0!dropout_42/dropout/Const:output:0*
T0*4
_output_shapes"
 :??????????????????@2
dropout_42/dropout/Mul?
dropout_42/dropout/ShapeShape0multi_head_attention_8/dense_99/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_42/dropout/Shape?
/dropout_42/dropout/random_uniform/RandomUniformRandomUniform!dropout_42/dropout/Shape:output:0*
T0*4
_output_shapes"
 :??????????????????@*
dtype021
/dropout_42/dropout/random_uniform/RandomUniform?
!dropout_42/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2#
!dropout_42/dropout/GreaterEqual/y?
dropout_42/dropout/GreaterEqualGreaterEqual8dropout_42/dropout/random_uniform/RandomUniform:output:0*dropout_42/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????@2!
dropout_42/dropout/GreaterEqual?
dropout_42/dropout/CastCast#dropout_42/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????@2
dropout_42/dropout/Cast?
dropout_42/dropout/Mul_1Muldropout_42/dropout/Mul:z:0dropout_42/dropout/Cast:y:0*
T0*4
_output_shapes"
 :??????????????????@2
dropout_42/dropout/Mul_1o
addAddV2inputsdropout_42/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????d@2
add?
5layer_normalization_26/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_26/moments/mean/reduction_indices?
#layer_normalization_26/moments/meanMeanadd:z:0>layer_normalization_26/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????d*
	keep_dims(2%
#layer_normalization_26/moments/mean?
+layer_normalization_26/moments/StopGradientStopGradient,layer_normalization_26/moments/mean:output:0*
T0*+
_output_shapes
:?????????d2-
+layer_normalization_26/moments/StopGradient?
0layer_normalization_26/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_26/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????d@22
0layer_normalization_26/moments/SquaredDifference?
9layer_normalization_26/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_26/moments/variance/reduction_indices?
'layer_normalization_26/moments/varianceMean4layer_normalization_26/moments/SquaredDifference:z:0Blayer_normalization_26/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????d*
	keep_dims(2)
'layer_normalization_26/moments/variance?
&layer_normalization_26/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52(
&layer_normalization_26/batchnorm/add/y?
$layer_normalization_26/batchnorm/addAddV20layer_normalization_26/moments/variance:output:0/layer_normalization_26/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????d2&
$layer_normalization_26/batchnorm/add?
&layer_normalization_26/batchnorm/RsqrtRsqrt(layer_normalization_26/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????d2(
&layer_normalization_26/batchnorm/Rsqrt?
3layer_normalization_26/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_26_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype025
3layer_normalization_26/batchnorm/mul/ReadVariableOp?
$layer_normalization_26/batchnorm/mulMul*layer_normalization_26/batchnorm/Rsqrt:y:0;layer_normalization_26/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@2&
$layer_normalization_26/batchnorm/mul?
&layer_normalization_26/batchnorm/mul_1Muladd:z:0(layer_normalization_26/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????d@2(
&layer_normalization_26/batchnorm/mul_1?
&layer_normalization_26/batchnorm/mul_2Mul,layer_normalization_26/moments/mean:output:0(layer_normalization_26/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????d@2(
&layer_normalization_26/batchnorm/mul_2?
/layer_normalization_26/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_26_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype021
/layer_normalization_26/batchnorm/ReadVariableOp?
$layer_normalization_26/batchnorm/subSub7layer_normalization_26/batchnorm/ReadVariableOp:value:0*layer_normalization_26/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????d@2&
$layer_normalization_26/batchnorm/sub?
&layer_normalization_26/batchnorm/add_1AddV2*layer_normalization_26/batchnorm/mul_1:z:0(layer_normalization_26/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????d@2(
&layer_normalization_26/batchnorm/add_1?
/sequential_8/dense_100/Tensordot/ReadVariableOpReadVariableOp8sequential_8_dense_100_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype021
/sequential_8/dense_100/Tensordot/ReadVariableOp?
%sequential_8/dense_100/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_8/dense_100/Tensordot/axes?
%sequential_8/dense_100/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_8/dense_100/Tensordot/free?
&sequential_8/dense_100/Tensordot/ShapeShape*layer_normalization_26/batchnorm/add_1:z:0*
T0*
_output_shapes
:2(
&sequential_8/dense_100/Tensordot/Shape?
.sequential_8/dense_100/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_8/dense_100/Tensordot/GatherV2/axis?
)sequential_8/dense_100/Tensordot/GatherV2GatherV2/sequential_8/dense_100/Tensordot/Shape:output:0.sequential_8/dense_100/Tensordot/free:output:07sequential_8/dense_100/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_8/dense_100/Tensordot/GatherV2?
0sequential_8/dense_100/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_8/dense_100/Tensordot/GatherV2_1/axis?
+sequential_8/dense_100/Tensordot/GatherV2_1GatherV2/sequential_8/dense_100/Tensordot/Shape:output:0.sequential_8/dense_100/Tensordot/axes:output:09sequential_8/dense_100/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_8/dense_100/Tensordot/GatherV2_1?
&sequential_8/dense_100/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_8/dense_100/Tensordot/Const?
%sequential_8/dense_100/Tensordot/ProdProd2sequential_8/dense_100/Tensordot/GatherV2:output:0/sequential_8/dense_100/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_8/dense_100/Tensordot/Prod?
(sequential_8/dense_100/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_8/dense_100/Tensordot/Const_1?
'sequential_8/dense_100/Tensordot/Prod_1Prod4sequential_8/dense_100/Tensordot/GatherV2_1:output:01sequential_8/dense_100/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_8/dense_100/Tensordot/Prod_1?
,sequential_8/dense_100/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_8/dense_100/Tensordot/concat/axis?
'sequential_8/dense_100/Tensordot/concatConcatV2.sequential_8/dense_100/Tensordot/free:output:0.sequential_8/dense_100/Tensordot/axes:output:05sequential_8/dense_100/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_8/dense_100/Tensordot/concat?
&sequential_8/dense_100/Tensordot/stackPack.sequential_8/dense_100/Tensordot/Prod:output:00sequential_8/dense_100/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_8/dense_100/Tensordot/stack?
*sequential_8/dense_100/Tensordot/transpose	Transpose*layer_normalization_26/batchnorm/add_1:z:00sequential_8/dense_100/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????d@2,
*sequential_8/dense_100/Tensordot/transpose?
(sequential_8/dense_100/Tensordot/ReshapeReshape.sequential_8/dense_100/Tensordot/transpose:y:0/sequential_8/dense_100/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(sequential_8/dense_100/Tensordot/Reshape?
'sequential_8/dense_100/Tensordot/MatMulMatMul1sequential_8/dense_100/Tensordot/Reshape:output:07sequential_8/dense_100/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2)
'sequential_8/dense_100/Tensordot/MatMul?
(sequential_8/dense_100/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_8/dense_100/Tensordot/Const_2?
.sequential_8/dense_100/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_8/dense_100/Tensordot/concat_1/axis?
)sequential_8/dense_100/Tensordot/concat_1ConcatV22sequential_8/dense_100/Tensordot/GatherV2:output:01sequential_8/dense_100/Tensordot/Const_2:output:07sequential_8/dense_100/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_8/dense_100/Tensordot/concat_1?
 sequential_8/dense_100/TensordotReshape1sequential_8/dense_100/Tensordot/MatMul:product:02sequential_8/dense_100/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????d 2"
 sequential_8/dense_100/Tensordot?
-sequential_8/dense_100/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_dense_100_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_8/dense_100/BiasAdd/ReadVariableOp?
sequential_8/dense_100/BiasAddBiasAdd)sequential_8/dense_100/Tensordot:output:05sequential_8/dense_100/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d 2 
sequential_8/dense_100/BiasAdd?
sequential_8/dense_100/ReluRelu'sequential_8/dense_100/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d 2
sequential_8/dense_100/Relu?
/sequential_8/dense_101/Tensordot/ReadVariableOpReadVariableOp8sequential_8_dense_101_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype021
/sequential_8/dense_101/Tensordot/ReadVariableOp?
%sequential_8/dense_101/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_8/dense_101/Tensordot/axes?
%sequential_8/dense_101/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_8/dense_101/Tensordot/free?
&sequential_8/dense_101/Tensordot/ShapeShape)sequential_8/dense_100/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_8/dense_101/Tensordot/Shape?
.sequential_8/dense_101/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_8/dense_101/Tensordot/GatherV2/axis?
)sequential_8/dense_101/Tensordot/GatherV2GatherV2/sequential_8/dense_101/Tensordot/Shape:output:0.sequential_8/dense_101/Tensordot/free:output:07sequential_8/dense_101/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_8/dense_101/Tensordot/GatherV2?
0sequential_8/dense_101/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_8/dense_101/Tensordot/GatherV2_1/axis?
+sequential_8/dense_101/Tensordot/GatherV2_1GatherV2/sequential_8/dense_101/Tensordot/Shape:output:0.sequential_8/dense_101/Tensordot/axes:output:09sequential_8/dense_101/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_8/dense_101/Tensordot/GatherV2_1?
&sequential_8/dense_101/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_8/dense_101/Tensordot/Const?
%sequential_8/dense_101/Tensordot/ProdProd2sequential_8/dense_101/Tensordot/GatherV2:output:0/sequential_8/dense_101/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_8/dense_101/Tensordot/Prod?
(sequential_8/dense_101/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_8/dense_101/Tensordot/Const_1?
'sequential_8/dense_101/Tensordot/Prod_1Prod4sequential_8/dense_101/Tensordot/GatherV2_1:output:01sequential_8/dense_101/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_8/dense_101/Tensordot/Prod_1?
,sequential_8/dense_101/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_8/dense_101/Tensordot/concat/axis?
'sequential_8/dense_101/Tensordot/concatConcatV2.sequential_8/dense_101/Tensordot/free:output:0.sequential_8/dense_101/Tensordot/axes:output:05sequential_8/dense_101/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_8/dense_101/Tensordot/concat?
&sequential_8/dense_101/Tensordot/stackPack.sequential_8/dense_101/Tensordot/Prod:output:00sequential_8/dense_101/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_8/dense_101/Tensordot/stack?
*sequential_8/dense_101/Tensordot/transpose	Transpose)sequential_8/dense_100/Relu:activations:00sequential_8/dense_101/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????d 2,
*sequential_8/dense_101/Tensordot/transpose?
(sequential_8/dense_101/Tensordot/ReshapeReshape.sequential_8/dense_101/Tensordot/transpose:y:0/sequential_8/dense_101/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(sequential_8/dense_101/Tensordot/Reshape?
'sequential_8/dense_101/Tensordot/MatMulMatMul1sequential_8/dense_101/Tensordot/Reshape:output:07sequential_8/dense_101/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2)
'sequential_8/dense_101/Tensordot/MatMul?
(sequential_8/dense_101/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2*
(sequential_8/dense_101/Tensordot/Const_2?
.sequential_8/dense_101/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_8/dense_101/Tensordot/concat_1/axis?
)sequential_8/dense_101/Tensordot/concat_1ConcatV22sequential_8/dense_101/Tensordot/GatherV2:output:01sequential_8/dense_101/Tensordot/Const_2:output:07sequential_8/dense_101/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_8/dense_101/Tensordot/concat_1?
 sequential_8/dense_101/TensordotReshape1sequential_8/dense_101/Tensordot/MatMul:product:02sequential_8/dense_101/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????d@2"
 sequential_8/dense_101/Tensordot?
-sequential_8/dense_101/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_dense_101_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-sequential_8/dense_101/BiasAdd/ReadVariableOp?
sequential_8/dense_101/BiasAddBiasAdd)sequential_8/dense_101/Tensordot:output:05sequential_8/dense_101/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@2 
sequential_8/dense_101/BiasAddy
dropout_43/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_43/dropout/Const?
dropout_43/dropout/MulMul'sequential_8/dense_101/BiasAdd:output:0!dropout_43/dropout/Const:output:0*
T0*+
_output_shapes
:?????????d@2
dropout_43/dropout/Mul?
dropout_43/dropout/ShapeShape'sequential_8/dense_101/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_43/dropout/Shape?
/dropout_43/dropout/random_uniform/RandomUniformRandomUniform!dropout_43/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????d@*
dtype021
/dropout_43/dropout/random_uniform/RandomUniform?
!dropout_43/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2#
!dropout_43/dropout/GreaterEqual/y?
dropout_43/dropout/GreaterEqualGreaterEqual8dropout_43/dropout/random_uniform/RandomUniform:output:0*dropout_43/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????d@2!
dropout_43/dropout/GreaterEqual?
dropout_43/dropout/CastCast#dropout_43/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????d@2
dropout_43/dropout/Cast?
dropout_43/dropout/Mul_1Muldropout_43/dropout/Mul:z:0dropout_43/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????d@2
dropout_43/dropout/Mul_1?
add_1AddV2*layer_normalization_26/batchnorm/add_1:z:0dropout_43/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????d@2
add_1?
5layer_normalization_27/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_27/moments/mean/reduction_indices?
#layer_normalization_27/moments/meanMean	add_1:z:0>layer_normalization_27/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????d*
	keep_dims(2%
#layer_normalization_27/moments/mean?
+layer_normalization_27/moments/StopGradientStopGradient,layer_normalization_27/moments/mean:output:0*
T0*+
_output_shapes
:?????????d2-
+layer_normalization_27/moments/StopGradient?
0layer_normalization_27/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_27/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????d@22
0layer_normalization_27/moments/SquaredDifference?
9layer_normalization_27/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_27/moments/variance/reduction_indices?
'layer_normalization_27/moments/varianceMean4layer_normalization_27/moments/SquaredDifference:z:0Blayer_normalization_27/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????d*
	keep_dims(2)
'layer_normalization_27/moments/variance?
&layer_normalization_27/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52(
&layer_normalization_27/batchnorm/add/y?
$layer_normalization_27/batchnorm/addAddV20layer_normalization_27/moments/variance:output:0/layer_normalization_27/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????d2&
$layer_normalization_27/batchnorm/add?
&layer_normalization_27/batchnorm/RsqrtRsqrt(layer_normalization_27/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????d2(
&layer_normalization_27/batchnorm/Rsqrt?
3layer_normalization_27/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_27_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype025
3layer_normalization_27/batchnorm/mul/ReadVariableOp?
$layer_normalization_27/batchnorm/mulMul*layer_normalization_27/batchnorm/Rsqrt:y:0;layer_normalization_27/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@2&
$layer_normalization_27/batchnorm/mul?
&layer_normalization_27/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_27/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????d@2(
&layer_normalization_27/batchnorm/mul_1?
&layer_normalization_27/batchnorm/mul_2Mul,layer_normalization_27/moments/mean:output:0(layer_normalization_27/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????d@2(
&layer_normalization_27/batchnorm/mul_2?
/layer_normalization_27/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_27_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype021
/layer_normalization_27/batchnorm/ReadVariableOp?
$layer_normalization_27/batchnorm/subSub7layer_normalization_27/batchnorm/ReadVariableOp:value:0*layer_normalization_27/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????d@2&
$layer_normalization_27/batchnorm/sub?
&layer_normalization_27/batchnorm/add_1AddV2*layer_normalization_27/batchnorm/mul_1:z:0(layer_normalization_27/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????d@2(
&layer_normalization_27/batchnorm/add_1?
IdentityIdentity*layer_normalization_27/batchnorm/add_1:z:00^layer_normalization_26/batchnorm/ReadVariableOp4^layer_normalization_26/batchnorm/mul/ReadVariableOp0^layer_normalization_27/batchnorm/ReadVariableOp4^layer_normalization_27/batchnorm/mul/ReadVariableOp7^multi_head_attention_8/dense_96/BiasAdd/ReadVariableOp9^multi_head_attention_8/dense_96/Tensordot/ReadVariableOp7^multi_head_attention_8/dense_97/BiasAdd/ReadVariableOp9^multi_head_attention_8/dense_97/Tensordot/ReadVariableOp7^multi_head_attention_8/dense_98/BiasAdd/ReadVariableOp9^multi_head_attention_8/dense_98/Tensordot/ReadVariableOp7^multi_head_attention_8/dense_99/BiasAdd/ReadVariableOp9^multi_head_attention_8/dense_99/Tensordot/ReadVariableOp.^sequential_8/dense_100/BiasAdd/ReadVariableOp0^sequential_8/dense_100/Tensordot/ReadVariableOp.^sequential_8/dense_101/BiasAdd/ReadVariableOp0^sequential_8/dense_101/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????d@: : : : : : : : : : : : : : : : 2b
/layer_normalization_26/batchnorm/ReadVariableOp/layer_normalization_26/batchnorm/ReadVariableOp2j
3layer_normalization_26/batchnorm/mul/ReadVariableOp3layer_normalization_26/batchnorm/mul/ReadVariableOp2b
/layer_normalization_27/batchnorm/ReadVariableOp/layer_normalization_27/batchnorm/ReadVariableOp2j
3layer_normalization_27/batchnorm/mul/ReadVariableOp3layer_normalization_27/batchnorm/mul/ReadVariableOp2p
6multi_head_attention_8/dense_96/BiasAdd/ReadVariableOp6multi_head_attention_8/dense_96/BiasAdd/ReadVariableOp2t
8multi_head_attention_8/dense_96/Tensordot/ReadVariableOp8multi_head_attention_8/dense_96/Tensordot/ReadVariableOp2p
6multi_head_attention_8/dense_97/BiasAdd/ReadVariableOp6multi_head_attention_8/dense_97/BiasAdd/ReadVariableOp2t
8multi_head_attention_8/dense_97/Tensordot/ReadVariableOp8multi_head_attention_8/dense_97/Tensordot/ReadVariableOp2p
6multi_head_attention_8/dense_98/BiasAdd/ReadVariableOp6multi_head_attention_8/dense_98/BiasAdd/ReadVariableOp2t
8multi_head_attention_8/dense_98/Tensordot/ReadVariableOp8multi_head_attention_8/dense_98/Tensordot/ReadVariableOp2p
6multi_head_attention_8/dense_99/BiasAdd/ReadVariableOp6multi_head_attention_8/dense_99/BiasAdd/ReadVariableOp2t
8multi_head_attention_8/dense_99/Tensordot/ReadVariableOp8multi_head_attention_8/dense_99/Tensordot/ReadVariableOp2^
-sequential_8/dense_100/BiasAdd/ReadVariableOp-sequential_8/dense_100/BiasAdd/ReadVariableOp2b
/sequential_8/dense_100/Tensordot/ReadVariableOp/sequential_8/dense_100/Tensordot/ReadVariableOp2^
-sequential_8/dense_101/BiasAdd/ReadVariableOp-sequential_8/dense_101/BiasAdd/ReadVariableOp2b
/sequential_8/dense_101/Tensordot/ReadVariableOp/sequential_8/dense_101/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????d@
 
_user_specified_nameinputs
ߣ
?
B__inference_model_8_layer_call_and_return_conditional_losses_92919

inputsT
Btoken_and_position_embedding_9_embedding_21_embedding_lookup_92620:d@U
Btoken_and_position_embedding_9_embedding_20_embedding_lookup_92626:	?@@g
Utransformer_block_8_multi_head_attention_8_dense_96_tensordot_readvariableop_resource:@@a
Stransformer_block_8_multi_head_attention_8_dense_96_biasadd_readvariableop_resource:@g
Utransformer_block_8_multi_head_attention_8_dense_97_tensordot_readvariableop_resource:@@a
Stransformer_block_8_multi_head_attention_8_dense_97_biasadd_readvariableop_resource:@g
Utransformer_block_8_multi_head_attention_8_dense_98_tensordot_readvariableop_resource:@@a
Stransformer_block_8_multi_head_attention_8_dense_98_biasadd_readvariableop_resource:@g
Utransformer_block_8_multi_head_attention_8_dense_99_tensordot_readvariableop_resource:@@a
Stransformer_block_8_multi_head_attention_8_dense_99_biasadd_readvariableop_resource:@^
Ptransformer_block_8_layer_normalization_26_batchnorm_mul_readvariableop_resource:@Z
Ltransformer_block_8_layer_normalization_26_batchnorm_readvariableop_resource:@^
Ltransformer_block_8_sequential_8_dense_100_tensordot_readvariableop_resource:@ X
Jtransformer_block_8_sequential_8_dense_100_biasadd_readvariableop_resource: ^
Ltransformer_block_8_sequential_8_dense_101_tensordot_readvariableop_resource: @X
Jtransformer_block_8_sequential_8_dense_101_biasadd_readvariableop_resource:@^
Ptransformer_block_8_layer_normalization_27_batchnorm_mul_readvariableop_resource:@Z
Ltransformer_block_8_layer_normalization_27_batchnorm_readvariableop_resource:@:
(dense_102_matmul_readvariableop_resource:@7
)dense_102_biasadd_readvariableop_resource::
(dense_103_matmul_readvariableop_resource:7
)dense_103_biasadd_readvariableop_resource:
identity?? dense_102/BiasAdd/ReadVariableOp?dense_102/MatMul/ReadVariableOp? dense_103/BiasAdd/ReadVariableOp?dense_103/MatMul/ReadVariableOp?<token_and_position_embedding_9/embedding_20/embedding_lookup?<token_and_position_embedding_9/embedding_21/embedding_lookup?Ctransformer_block_8/layer_normalization_26/batchnorm/ReadVariableOp?Gtransformer_block_8/layer_normalization_26/batchnorm/mul/ReadVariableOp?Ctransformer_block_8/layer_normalization_27/batchnorm/ReadVariableOp?Gtransformer_block_8/layer_normalization_27/batchnorm/mul/ReadVariableOp?Jtransformer_block_8/multi_head_attention_8/dense_96/BiasAdd/ReadVariableOp?Ltransformer_block_8/multi_head_attention_8/dense_96/Tensordot/ReadVariableOp?Jtransformer_block_8/multi_head_attention_8/dense_97/BiasAdd/ReadVariableOp?Ltransformer_block_8/multi_head_attention_8/dense_97/Tensordot/ReadVariableOp?Jtransformer_block_8/multi_head_attention_8/dense_98/BiasAdd/ReadVariableOp?Ltransformer_block_8/multi_head_attention_8/dense_98/Tensordot/ReadVariableOp?Jtransformer_block_8/multi_head_attention_8/dense_99/BiasAdd/ReadVariableOp?Ltransformer_block_8/multi_head_attention_8/dense_99/Tensordot/ReadVariableOp?Atransformer_block_8/sequential_8/dense_100/BiasAdd/ReadVariableOp?Ctransformer_block_8/sequential_8/dense_100/Tensordot/ReadVariableOp?Atransformer_block_8/sequential_8/dense_101/BiasAdd/ReadVariableOp?Ctransformer_block_8/sequential_8/dense_101/Tensordot/ReadVariableOp?
$token_and_position_embedding_9/ShapeShapeinputs*
T0*
_output_shapes
:2&
$token_and_position_embedding_9/Shape?
2token_and_position_embedding_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????24
2token_and_position_embedding_9/strided_slice/stack?
4token_and_position_embedding_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 26
4token_and_position_embedding_9/strided_slice/stack_1?
4token_and_position_embedding_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4token_and_position_embedding_9/strided_slice/stack_2?
,token_and_position_embedding_9/strided_sliceStridedSlice-token_and_position_embedding_9/Shape:output:0;token_and_position_embedding_9/strided_slice/stack:output:0=token_and_position_embedding_9/strided_slice/stack_1:output:0=token_and_position_embedding_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,token_and_position_embedding_9/strided_slice?
*token_and_position_embedding_9/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2,
*token_and_position_embedding_9/range/start?
*token_and_position_embedding_9/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2,
*token_and_position_embedding_9/range/delta?
$token_and_position_embedding_9/rangeRange3token_and_position_embedding_9/range/start:output:05token_and_position_embedding_9/strided_slice:output:03token_and_position_embedding_9/range/delta:output:0*#
_output_shapes
:?????????2&
$token_and_position_embedding_9/range?
<token_and_position_embedding_9/embedding_21/embedding_lookupResourceGatherBtoken_and_position_embedding_9_embedding_21_embedding_lookup_92620-token_and_position_embedding_9/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*U
_classK
IGloc:@token_and_position_embedding_9/embedding_21/embedding_lookup/92620*'
_output_shapes
:?????????@*
dtype02>
<token_and_position_embedding_9/embedding_21/embedding_lookup?
Etoken_and_position_embedding_9/embedding_21/embedding_lookup/IdentityIdentityEtoken_and_position_embedding_9/embedding_21/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@token_and_position_embedding_9/embedding_21/embedding_lookup/92620*'
_output_shapes
:?????????@2G
Etoken_and_position_embedding_9/embedding_21/embedding_lookup/Identity?
Gtoken_and_position_embedding_9/embedding_21/embedding_lookup/Identity_1IdentityNtoken_and_position_embedding_9/embedding_21/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????@2I
Gtoken_and_position_embedding_9/embedding_21/embedding_lookup/Identity_1?
0token_and_position_embedding_9/embedding_20/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????d22
0token_and_position_embedding_9/embedding_20/Cast?
<token_and_position_embedding_9/embedding_20/embedding_lookupResourceGatherBtoken_and_position_embedding_9_embedding_20_embedding_lookup_926264token_and_position_embedding_9/embedding_20/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*U
_classK
IGloc:@token_and_position_embedding_9/embedding_20/embedding_lookup/92626*+
_output_shapes
:?????????d@*
dtype02>
<token_and_position_embedding_9/embedding_20/embedding_lookup?
Etoken_and_position_embedding_9/embedding_20/embedding_lookup/IdentityIdentityEtoken_and_position_embedding_9/embedding_20/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@token_and_position_embedding_9/embedding_20/embedding_lookup/92626*+
_output_shapes
:?????????d@2G
Etoken_and_position_embedding_9/embedding_20/embedding_lookup/Identity?
Gtoken_and_position_embedding_9/embedding_20/embedding_lookup/Identity_1IdentityNtoken_and_position_embedding_9/embedding_20/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????d@2I
Gtoken_and_position_embedding_9/embedding_20/embedding_lookup/Identity_1?
"token_and_position_embedding_9/addAddV2Ptoken_and_position_embedding_9/embedding_20/embedding_lookup/Identity_1:output:0Ptoken_and_position_embedding_9/embedding_21/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:?????????d@2$
"token_and_position_embedding_9/add?
0transformer_block_8/multi_head_attention_8/ShapeShape&token_and_position_embedding_9/add:z:0*
T0*
_output_shapes
:22
0transformer_block_8/multi_head_attention_8/Shape?
>transformer_block_8/multi_head_attention_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>transformer_block_8/multi_head_attention_8/strided_slice/stack?
@transformer_block_8/multi_head_attention_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@transformer_block_8/multi_head_attention_8/strided_slice/stack_1?
@transformer_block_8/multi_head_attention_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@transformer_block_8/multi_head_attention_8/strided_slice/stack_2?
8transformer_block_8/multi_head_attention_8/strided_sliceStridedSlice9transformer_block_8/multi_head_attention_8/Shape:output:0Gtransformer_block_8/multi_head_attention_8/strided_slice/stack:output:0Itransformer_block_8/multi_head_attention_8/strided_slice/stack_1:output:0Itransformer_block_8/multi_head_attention_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2:
8transformer_block_8/multi_head_attention_8/strided_slice?
Ltransformer_block_8/multi_head_attention_8/dense_96/Tensordot/ReadVariableOpReadVariableOpUtransformer_block_8_multi_head_attention_8_dense_96_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02N
Ltransformer_block_8/multi_head_attention_8/dense_96/Tensordot/ReadVariableOp?
Btransformer_block_8/multi_head_attention_8/dense_96/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2D
Btransformer_block_8/multi_head_attention_8/dense_96/Tensordot/axes?
Btransformer_block_8/multi_head_attention_8/dense_96/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2D
Btransformer_block_8/multi_head_attention_8/dense_96/Tensordot/free?
Ctransformer_block_8/multi_head_attention_8/dense_96/Tensordot/ShapeShape&token_and_position_embedding_9/add:z:0*
T0*
_output_shapes
:2E
Ctransformer_block_8/multi_head_attention_8/dense_96/Tensordot/Shape?
Ktransformer_block_8/multi_head_attention_8/dense_96/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Ktransformer_block_8/multi_head_attention_8/dense_96/Tensordot/GatherV2/axis?
Ftransformer_block_8/multi_head_attention_8/dense_96/Tensordot/GatherV2GatherV2Ltransformer_block_8/multi_head_attention_8/dense_96/Tensordot/Shape:output:0Ktransformer_block_8/multi_head_attention_8/dense_96/Tensordot/free:output:0Ttransformer_block_8/multi_head_attention_8/dense_96/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2H
Ftransformer_block_8/multi_head_attention_8/dense_96/Tensordot/GatherV2?
Mtransformer_block_8/multi_head_attention_8/dense_96/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2O
Mtransformer_block_8/multi_head_attention_8/dense_96/Tensordot/GatherV2_1/axis?
Htransformer_block_8/multi_head_attention_8/dense_96/Tensordot/GatherV2_1GatherV2Ltransformer_block_8/multi_head_attention_8/dense_96/Tensordot/Shape:output:0Ktransformer_block_8/multi_head_attention_8/dense_96/Tensordot/axes:output:0Vtransformer_block_8/multi_head_attention_8/dense_96/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2J
Htransformer_block_8/multi_head_attention_8/dense_96/Tensordot/GatherV2_1?
Ctransformer_block_8/multi_head_attention_8/dense_96/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2E
Ctransformer_block_8/multi_head_attention_8/dense_96/Tensordot/Const?
Btransformer_block_8/multi_head_attention_8/dense_96/Tensordot/ProdProdOtransformer_block_8/multi_head_attention_8/dense_96/Tensordot/GatherV2:output:0Ltransformer_block_8/multi_head_attention_8/dense_96/Tensordot/Const:output:0*
T0*
_output_shapes
: 2D
Btransformer_block_8/multi_head_attention_8/dense_96/Tensordot/Prod?
Etransformer_block_8/multi_head_attention_8/dense_96/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2G
Etransformer_block_8/multi_head_attention_8/dense_96/Tensordot/Const_1?
Dtransformer_block_8/multi_head_attention_8/dense_96/Tensordot/Prod_1ProdQtransformer_block_8/multi_head_attention_8/dense_96/Tensordot/GatherV2_1:output:0Ntransformer_block_8/multi_head_attention_8/dense_96/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2F
Dtransformer_block_8/multi_head_attention_8/dense_96/Tensordot/Prod_1?
Itransformer_block_8/multi_head_attention_8/dense_96/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Itransformer_block_8/multi_head_attention_8/dense_96/Tensordot/concat/axis?
Dtransformer_block_8/multi_head_attention_8/dense_96/Tensordot/concatConcatV2Ktransformer_block_8/multi_head_attention_8/dense_96/Tensordot/free:output:0Ktransformer_block_8/multi_head_attention_8/dense_96/Tensordot/axes:output:0Rtransformer_block_8/multi_head_attention_8/dense_96/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2F
Dtransformer_block_8/multi_head_attention_8/dense_96/Tensordot/concat?
Ctransformer_block_8/multi_head_attention_8/dense_96/Tensordot/stackPackKtransformer_block_8/multi_head_attention_8/dense_96/Tensordot/Prod:output:0Mtransformer_block_8/multi_head_attention_8/dense_96/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2E
Ctransformer_block_8/multi_head_attention_8/dense_96/Tensordot/stack?
Gtransformer_block_8/multi_head_attention_8/dense_96/Tensordot/transpose	Transpose&token_and_position_embedding_9/add:z:0Mtransformer_block_8/multi_head_attention_8/dense_96/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????d@2I
Gtransformer_block_8/multi_head_attention_8/dense_96/Tensordot/transpose?
Etransformer_block_8/multi_head_attention_8/dense_96/Tensordot/ReshapeReshapeKtransformer_block_8/multi_head_attention_8/dense_96/Tensordot/transpose:y:0Ltransformer_block_8/multi_head_attention_8/dense_96/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2G
Etransformer_block_8/multi_head_attention_8/dense_96/Tensordot/Reshape?
Dtransformer_block_8/multi_head_attention_8/dense_96/Tensordot/MatMulMatMulNtransformer_block_8/multi_head_attention_8/dense_96/Tensordot/Reshape:output:0Ttransformer_block_8/multi_head_attention_8/dense_96/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2F
Dtransformer_block_8/multi_head_attention_8/dense_96/Tensordot/MatMul?
Etransformer_block_8/multi_head_attention_8/dense_96/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2G
Etransformer_block_8/multi_head_attention_8/dense_96/Tensordot/Const_2?
Ktransformer_block_8/multi_head_attention_8/dense_96/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Ktransformer_block_8/multi_head_attention_8/dense_96/Tensordot/concat_1/axis?
Ftransformer_block_8/multi_head_attention_8/dense_96/Tensordot/concat_1ConcatV2Otransformer_block_8/multi_head_attention_8/dense_96/Tensordot/GatherV2:output:0Ntransformer_block_8/multi_head_attention_8/dense_96/Tensordot/Const_2:output:0Ttransformer_block_8/multi_head_attention_8/dense_96/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2H
Ftransformer_block_8/multi_head_attention_8/dense_96/Tensordot/concat_1?
=transformer_block_8/multi_head_attention_8/dense_96/TensordotReshapeNtransformer_block_8/multi_head_attention_8/dense_96/Tensordot/MatMul:product:0Otransformer_block_8/multi_head_attention_8/dense_96/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????d@2?
=transformer_block_8/multi_head_attention_8/dense_96/Tensordot?
Jtransformer_block_8/multi_head_attention_8/dense_96/BiasAdd/ReadVariableOpReadVariableOpStransformer_block_8_multi_head_attention_8_dense_96_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02L
Jtransformer_block_8/multi_head_attention_8/dense_96/BiasAdd/ReadVariableOp?
;transformer_block_8/multi_head_attention_8/dense_96/BiasAddBiasAddFtransformer_block_8/multi_head_attention_8/dense_96/Tensordot:output:0Rtransformer_block_8/multi_head_attention_8/dense_96/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@2=
;transformer_block_8/multi_head_attention_8/dense_96/BiasAdd?
Ltransformer_block_8/multi_head_attention_8/dense_97/Tensordot/ReadVariableOpReadVariableOpUtransformer_block_8_multi_head_attention_8_dense_97_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02N
Ltransformer_block_8/multi_head_attention_8/dense_97/Tensordot/ReadVariableOp?
Btransformer_block_8/multi_head_attention_8/dense_97/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2D
Btransformer_block_8/multi_head_attention_8/dense_97/Tensordot/axes?
Btransformer_block_8/multi_head_attention_8/dense_97/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2D
Btransformer_block_8/multi_head_attention_8/dense_97/Tensordot/free?
Ctransformer_block_8/multi_head_attention_8/dense_97/Tensordot/ShapeShape&token_and_position_embedding_9/add:z:0*
T0*
_output_shapes
:2E
Ctransformer_block_8/multi_head_attention_8/dense_97/Tensordot/Shape?
Ktransformer_block_8/multi_head_attention_8/dense_97/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Ktransformer_block_8/multi_head_attention_8/dense_97/Tensordot/GatherV2/axis?
Ftransformer_block_8/multi_head_attention_8/dense_97/Tensordot/GatherV2GatherV2Ltransformer_block_8/multi_head_attention_8/dense_97/Tensordot/Shape:output:0Ktransformer_block_8/multi_head_attention_8/dense_97/Tensordot/free:output:0Ttransformer_block_8/multi_head_attention_8/dense_97/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2H
Ftransformer_block_8/multi_head_attention_8/dense_97/Tensordot/GatherV2?
Mtransformer_block_8/multi_head_attention_8/dense_97/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2O
Mtransformer_block_8/multi_head_attention_8/dense_97/Tensordot/GatherV2_1/axis?
Htransformer_block_8/multi_head_attention_8/dense_97/Tensordot/GatherV2_1GatherV2Ltransformer_block_8/multi_head_attention_8/dense_97/Tensordot/Shape:output:0Ktransformer_block_8/multi_head_attention_8/dense_97/Tensordot/axes:output:0Vtransformer_block_8/multi_head_attention_8/dense_97/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2J
Htransformer_block_8/multi_head_attention_8/dense_97/Tensordot/GatherV2_1?
Ctransformer_block_8/multi_head_attention_8/dense_97/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2E
Ctransformer_block_8/multi_head_attention_8/dense_97/Tensordot/Const?
Btransformer_block_8/multi_head_attention_8/dense_97/Tensordot/ProdProdOtransformer_block_8/multi_head_attention_8/dense_97/Tensordot/GatherV2:output:0Ltransformer_block_8/multi_head_attention_8/dense_97/Tensordot/Const:output:0*
T0*
_output_shapes
: 2D
Btransformer_block_8/multi_head_attention_8/dense_97/Tensordot/Prod?
Etransformer_block_8/multi_head_attention_8/dense_97/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2G
Etransformer_block_8/multi_head_attention_8/dense_97/Tensordot/Const_1?
Dtransformer_block_8/multi_head_attention_8/dense_97/Tensordot/Prod_1ProdQtransformer_block_8/multi_head_attention_8/dense_97/Tensordot/GatherV2_1:output:0Ntransformer_block_8/multi_head_attention_8/dense_97/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2F
Dtransformer_block_8/multi_head_attention_8/dense_97/Tensordot/Prod_1?
Itransformer_block_8/multi_head_attention_8/dense_97/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Itransformer_block_8/multi_head_attention_8/dense_97/Tensordot/concat/axis?
Dtransformer_block_8/multi_head_attention_8/dense_97/Tensordot/concatConcatV2Ktransformer_block_8/multi_head_attention_8/dense_97/Tensordot/free:output:0Ktransformer_block_8/multi_head_attention_8/dense_97/Tensordot/axes:output:0Rtransformer_block_8/multi_head_attention_8/dense_97/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2F
Dtransformer_block_8/multi_head_attention_8/dense_97/Tensordot/concat?
Ctransformer_block_8/multi_head_attention_8/dense_97/Tensordot/stackPackKtransformer_block_8/multi_head_attention_8/dense_97/Tensordot/Prod:output:0Mtransformer_block_8/multi_head_attention_8/dense_97/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2E
Ctransformer_block_8/multi_head_attention_8/dense_97/Tensordot/stack?
Gtransformer_block_8/multi_head_attention_8/dense_97/Tensordot/transpose	Transpose&token_and_position_embedding_9/add:z:0Mtransformer_block_8/multi_head_attention_8/dense_97/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????d@2I
Gtransformer_block_8/multi_head_attention_8/dense_97/Tensordot/transpose?
Etransformer_block_8/multi_head_attention_8/dense_97/Tensordot/ReshapeReshapeKtransformer_block_8/multi_head_attention_8/dense_97/Tensordot/transpose:y:0Ltransformer_block_8/multi_head_attention_8/dense_97/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2G
Etransformer_block_8/multi_head_attention_8/dense_97/Tensordot/Reshape?
Dtransformer_block_8/multi_head_attention_8/dense_97/Tensordot/MatMulMatMulNtransformer_block_8/multi_head_attention_8/dense_97/Tensordot/Reshape:output:0Ttransformer_block_8/multi_head_attention_8/dense_97/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2F
Dtransformer_block_8/multi_head_attention_8/dense_97/Tensordot/MatMul?
Etransformer_block_8/multi_head_attention_8/dense_97/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2G
Etransformer_block_8/multi_head_attention_8/dense_97/Tensordot/Const_2?
Ktransformer_block_8/multi_head_attention_8/dense_97/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Ktransformer_block_8/multi_head_attention_8/dense_97/Tensordot/concat_1/axis?
Ftransformer_block_8/multi_head_attention_8/dense_97/Tensordot/concat_1ConcatV2Otransformer_block_8/multi_head_attention_8/dense_97/Tensordot/GatherV2:output:0Ntransformer_block_8/multi_head_attention_8/dense_97/Tensordot/Const_2:output:0Ttransformer_block_8/multi_head_attention_8/dense_97/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2H
Ftransformer_block_8/multi_head_attention_8/dense_97/Tensordot/concat_1?
=transformer_block_8/multi_head_attention_8/dense_97/TensordotReshapeNtransformer_block_8/multi_head_attention_8/dense_97/Tensordot/MatMul:product:0Otransformer_block_8/multi_head_attention_8/dense_97/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????d@2?
=transformer_block_8/multi_head_attention_8/dense_97/Tensordot?
Jtransformer_block_8/multi_head_attention_8/dense_97/BiasAdd/ReadVariableOpReadVariableOpStransformer_block_8_multi_head_attention_8_dense_97_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02L
Jtransformer_block_8/multi_head_attention_8/dense_97/BiasAdd/ReadVariableOp?
;transformer_block_8/multi_head_attention_8/dense_97/BiasAddBiasAddFtransformer_block_8/multi_head_attention_8/dense_97/Tensordot:output:0Rtransformer_block_8/multi_head_attention_8/dense_97/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@2=
;transformer_block_8/multi_head_attention_8/dense_97/BiasAdd?
Ltransformer_block_8/multi_head_attention_8/dense_98/Tensordot/ReadVariableOpReadVariableOpUtransformer_block_8_multi_head_attention_8_dense_98_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02N
Ltransformer_block_8/multi_head_attention_8/dense_98/Tensordot/ReadVariableOp?
Btransformer_block_8/multi_head_attention_8/dense_98/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2D
Btransformer_block_8/multi_head_attention_8/dense_98/Tensordot/axes?
Btransformer_block_8/multi_head_attention_8/dense_98/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2D
Btransformer_block_8/multi_head_attention_8/dense_98/Tensordot/free?
Ctransformer_block_8/multi_head_attention_8/dense_98/Tensordot/ShapeShape&token_and_position_embedding_9/add:z:0*
T0*
_output_shapes
:2E
Ctransformer_block_8/multi_head_attention_8/dense_98/Tensordot/Shape?
Ktransformer_block_8/multi_head_attention_8/dense_98/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Ktransformer_block_8/multi_head_attention_8/dense_98/Tensordot/GatherV2/axis?
Ftransformer_block_8/multi_head_attention_8/dense_98/Tensordot/GatherV2GatherV2Ltransformer_block_8/multi_head_attention_8/dense_98/Tensordot/Shape:output:0Ktransformer_block_8/multi_head_attention_8/dense_98/Tensordot/free:output:0Ttransformer_block_8/multi_head_attention_8/dense_98/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2H
Ftransformer_block_8/multi_head_attention_8/dense_98/Tensordot/GatherV2?
Mtransformer_block_8/multi_head_attention_8/dense_98/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2O
Mtransformer_block_8/multi_head_attention_8/dense_98/Tensordot/GatherV2_1/axis?
Htransformer_block_8/multi_head_attention_8/dense_98/Tensordot/GatherV2_1GatherV2Ltransformer_block_8/multi_head_attention_8/dense_98/Tensordot/Shape:output:0Ktransformer_block_8/multi_head_attention_8/dense_98/Tensordot/axes:output:0Vtransformer_block_8/multi_head_attention_8/dense_98/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2J
Htransformer_block_8/multi_head_attention_8/dense_98/Tensordot/GatherV2_1?
Ctransformer_block_8/multi_head_attention_8/dense_98/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2E
Ctransformer_block_8/multi_head_attention_8/dense_98/Tensordot/Const?
Btransformer_block_8/multi_head_attention_8/dense_98/Tensordot/ProdProdOtransformer_block_8/multi_head_attention_8/dense_98/Tensordot/GatherV2:output:0Ltransformer_block_8/multi_head_attention_8/dense_98/Tensordot/Const:output:0*
T0*
_output_shapes
: 2D
Btransformer_block_8/multi_head_attention_8/dense_98/Tensordot/Prod?
Etransformer_block_8/multi_head_attention_8/dense_98/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2G
Etransformer_block_8/multi_head_attention_8/dense_98/Tensordot/Const_1?
Dtransformer_block_8/multi_head_attention_8/dense_98/Tensordot/Prod_1ProdQtransformer_block_8/multi_head_attention_8/dense_98/Tensordot/GatherV2_1:output:0Ntransformer_block_8/multi_head_attention_8/dense_98/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2F
Dtransformer_block_8/multi_head_attention_8/dense_98/Tensordot/Prod_1?
Itransformer_block_8/multi_head_attention_8/dense_98/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Itransformer_block_8/multi_head_attention_8/dense_98/Tensordot/concat/axis?
Dtransformer_block_8/multi_head_attention_8/dense_98/Tensordot/concatConcatV2Ktransformer_block_8/multi_head_attention_8/dense_98/Tensordot/free:output:0Ktransformer_block_8/multi_head_attention_8/dense_98/Tensordot/axes:output:0Rtransformer_block_8/multi_head_attention_8/dense_98/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2F
Dtransformer_block_8/multi_head_attention_8/dense_98/Tensordot/concat?
Ctransformer_block_8/multi_head_attention_8/dense_98/Tensordot/stackPackKtransformer_block_8/multi_head_attention_8/dense_98/Tensordot/Prod:output:0Mtransformer_block_8/multi_head_attention_8/dense_98/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2E
Ctransformer_block_8/multi_head_attention_8/dense_98/Tensordot/stack?
Gtransformer_block_8/multi_head_attention_8/dense_98/Tensordot/transpose	Transpose&token_and_position_embedding_9/add:z:0Mtransformer_block_8/multi_head_attention_8/dense_98/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????d@2I
Gtransformer_block_8/multi_head_attention_8/dense_98/Tensordot/transpose?
Etransformer_block_8/multi_head_attention_8/dense_98/Tensordot/ReshapeReshapeKtransformer_block_8/multi_head_attention_8/dense_98/Tensordot/transpose:y:0Ltransformer_block_8/multi_head_attention_8/dense_98/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2G
Etransformer_block_8/multi_head_attention_8/dense_98/Tensordot/Reshape?
Dtransformer_block_8/multi_head_attention_8/dense_98/Tensordot/MatMulMatMulNtransformer_block_8/multi_head_attention_8/dense_98/Tensordot/Reshape:output:0Ttransformer_block_8/multi_head_attention_8/dense_98/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2F
Dtransformer_block_8/multi_head_attention_8/dense_98/Tensordot/MatMul?
Etransformer_block_8/multi_head_attention_8/dense_98/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2G
Etransformer_block_8/multi_head_attention_8/dense_98/Tensordot/Const_2?
Ktransformer_block_8/multi_head_attention_8/dense_98/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Ktransformer_block_8/multi_head_attention_8/dense_98/Tensordot/concat_1/axis?
Ftransformer_block_8/multi_head_attention_8/dense_98/Tensordot/concat_1ConcatV2Otransformer_block_8/multi_head_attention_8/dense_98/Tensordot/GatherV2:output:0Ntransformer_block_8/multi_head_attention_8/dense_98/Tensordot/Const_2:output:0Ttransformer_block_8/multi_head_attention_8/dense_98/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2H
Ftransformer_block_8/multi_head_attention_8/dense_98/Tensordot/concat_1?
=transformer_block_8/multi_head_attention_8/dense_98/TensordotReshapeNtransformer_block_8/multi_head_attention_8/dense_98/Tensordot/MatMul:product:0Otransformer_block_8/multi_head_attention_8/dense_98/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????d@2?
=transformer_block_8/multi_head_attention_8/dense_98/Tensordot?
Jtransformer_block_8/multi_head_attention_8/dense_98/BiasAdd/ReadVariableOpReadVariableOpStransformer_block_8_multi_head_attention_8_dense_98_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02L
Jtransformer_block_8/multi_head_attention_8/dense_98/BiasAdd/ReadVariableOp?
;transformer_block_8/multi_head_attention_8/dense_98/BiasAddBiasAddFtransformer_block_8/multi_head_attention_8/dense_98/Tensordot:output:0Rtransformer_block_8/multi_head_attention_8/dense_98/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@2=
;transformer_block_8/multi_head_attention_8/dense_98/BiasAdd?
:transformer_block_8/multi_head_attention_8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2<
:transformer_block_8/multi_head_attention_8/Reshape/shape/1?
:transformer_block_8/multi_head_attention_8/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2<
:transformer_block_8/multi_head_attention_8/Reshape/shape/2?
:transformer_block_8/multi_head_attention_8/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2<
:transformer_block_8/multi_head_attention_8/Reshape/shape/3?
8transformer_block_8/multi_head_attention_8/Reshape/shapePackAtransformer_block_8/multi_head_attention_8/strided_slice:output:0Ctransformer_block_8/multi_head_attention_8/Reshape/shape/1:output:0Ctransformer_block_8/multi_head_attention_8/Reshape/shape/2:output:0Ctransformer_block_8/multi_head_attention_8/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block_8/multi_head_attention_8/Reshape/shape?
2transformer_block_8/multi_head_attention_8/ReshapeReshapeDtransformer_block_8/multi_head_attention_8/dense_96/BiasAdd:output:0Atransformer_block_8/multi_head_attention_8/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"?????????????????? 24
2transformer_block_8/multi_head_attention_8/Reshape?
9transformer_block_8/multi_head_attention_8/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2;
9transformer_block_8/multi_head_attention_8/transpose/perm?
4transformer_block_8/multi_head_attention_8/transpose	Transpose;transformer_block_8/multi_head_attention_8/Reshape:output:0Btransformer_block_8/multi_head_attention_8/transpose/perm:output:0*
T0*8
_output_shapes&
$:"?????????????????? 26
4transformer_block_8/multi_head_attention_8/transpose?
<transformer_block_8/multi_head_attention_8/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2>
<transformer_block_8/multi_head_attention_8/Reshape_1/shape/1?
<transformer_block_8/multi_head_attention_8/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2>
<transformer_block_8/multi_head_attention_8/Reshape_1/shape/2?
<transformer_block_8/multi_head_attention_8/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2>
<transformer_block_8/multi_head_attention_8/Reshape_1/shape/3?
:transformer_block_8/multi_head_attention_8/Reshape_1/shapePackAtransformer_block_8/multi_head_attention_8/strided_slice:output:0Etransformer_block_8/multi_head_attention_8/Reshape_1/shape/1:output:0Etransformer_block_8/multi_head_attention_8/Reshape_1/shape/2:output:0Etransformer_block_8/multi_head_attention_8/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_8/multi_head_attention_8/Reshape_1/shape?
4transformer_block_8/multi_head_attention_8/Reshape_1ReshapeDtransformer_block_8/multi_head_attention_8/dense_97/BiasAdd:output:0Ctransformer_block_8/multi_head_attention_8/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"?????????????????? 26
4transformer_block_8/multi_head_attention_8/Reshape_1?
;transformer_block_8/multi_head_attention_8/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2=
;transformer_block_8/multi_head_attention_8/transpose_1/perm?
6transformer_block_8/multi_head_attention_8/transpose_1	Transpose=transformer_block_8/multi_head_attention_8/Reshape_1:output:0Dtransformer_block_8/multi_head_attention_8/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"?????????????????? 28
6transformer_block_8/multi_head_attention_8/transpose_1?
<transformer_block_8/multi_head_attention_8/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2>
<transformer_block_8/multi_head_attention_8/Reshape_2/shape/1?
<transformer_block_8/multi_head_attention_8/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2>
<transformer_block_8/multi_head_attention_8/Reshape_2/shape/2?
<transformer_block_8/multi_head_attention_8/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2>
<transformer_block_8/multi_head_attention_8/Reshape_2/shape/3?
:transformer_block_8/multi_head_attention_8/Reshape_2/shapePackAtransformer_block_8/multi_head_attention_8/strided_slice:output:0Etransformer_block_8/multi_head_attention_8/Reshape_2/shape/1:output:0Etransformer_block_8/multi_head_attention_8/Reshape_2/shape/2:output:0Etransformer_block_8/multi_head_attention_8/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_8/multi_head_attention_8/Reshape_2/shape?
4transformer_block_8/multi_head_attention_8/Reshape_2ReshapeDtransformer_block_8/multi_head_attention_8/dense_98/BiasAdd:output:0Ctransformer_block_8/multi_head_attention_8/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"?????????????????? 26
4transformer_block_8/multi_head_attention_8/Reshape_2?
;transformer_block_8/multi_head_attention_8/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2=
;transformer_block_8/multi_head_attention_8/transpose_2/perm?
6transformer_block_8/multi_head_attention_8/transpose_2	Transpose=transformer_block_8/multi_head_attention_8/Reshape_2:output:0Dtransformer_block_8/multi_head_attention_8/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"?????????????????? 28
6transformer_block_8/multi_head_attention_8/transpose_2?
1transformer_block_8/multi_head_attention_8/MatMulBatchMatMulV28transformer_block_8/multi_head_attention_8/transpose:y:0:transformer_block_8/multi_head_attention_8/transpose_1:y:0*
T0*A
_output_shapes/
-:+???????????????????????????*
adj_y(23
1transformer_block_8/multi_head_attention_8/MatMul?
2transformer_block_8/multi_head_attention_8/Shape_1Shape:transformer_block_8/multi_head_attention_8/transpose_1:y:0*
T0*
_output_shapes
:24
2transformer_block_8/multi_head_attention_8/Shape_1?
@transformer_block_8/multi_head_attention_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2B
@transformer_block_8/multi_head_attention_8/strided_slice_1/stack?
Btransformer_block_8/multi_head_attention_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2D
Btransformer_block_8/multi_head_attention_8/strided_slice_1/stack_1?
Btransformer_block_8/multi_head_attention_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
Btransformer_block_8/multi_head_attention_8/strided_slice_1/stack_2?
:transformer_block_8/multi_head_attention_8/strided_slice_1StridedSlice;transformer_block_8/multi_head_attention_8/Shape_1:output:0Itransformer_block_8/multi_head_attention_8/strided_slice_1/stack:output:0Ktransformer_block_8/multi_head_attention_8/strided_slice_1/stack_1:output:0Ktransformer_block_8/multi_head_attention_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2<
:transformer_block_8/multi_head_attention_8/strided_slice_1?
/transformer_block_8/multi_head_attention_8/CastCastCtransformer_block_8/multi_head_attention_8/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 21
/transformer_block_8/multi_head_attention_8/Cast?
/transformer_block_8/multi_head_attention_8/SqrtSqrt3transformer_block_8/multi_head_attention_8/Cast:y:0*
T0*
_output_shapes
: 21
/transformer_block_8/multi_head_attention_8/Sqrt?
2transformer_block_8/multi_head_attention_8/truedivRealDiv:transformer_block_8/multi_head_attention_8/MatMul:output:03transformer_block_8/multi_head_attention_8/Sqrt:y:0*
T0*A
_output_shapes/
-:+???????????????????????????24
2transformer_block_8/multi_head_attention_8/truediv?
2transformer_block_8/multi_head_attention_8/SoftmaxSoftmax6transformer_block_8/multi_head_attention_8/truediv:z:0*
T0*A
_output_shapes/
-:+???????????????????????????24
2transformer_block_8/multi_head_attention_8/Softmax?
3transformer_block_8/multi_head_attention_8/MatMul_1BatchMatMulV2<transformer_block_8/multi_head_attention_8/Softmax:softmax:0:transformer_block_8/multi_head_attention_8/transpose_2:y:0*
T0*8
_output_shapes&
$:"?????????????????? 25
3transformer_block_8/multi_head_attention_8/MatMul_1?
;transformer_block_8/multi_head_attention_8/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             2=
;transformer_block_8/multi_head_attention_8/transpose_3/perm?
6transformer_block_8/multi_head_attention_8/transpose_3	Transpose<transformer_block_8/multi_head_attention_8/MatMul_1:output:0Dtransformer_block_8/multi_head_attention_8/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"?????????????????? 28
6transformer_block_8/multi_head_attention_8/transpose_3?
<transformer_block_8/multi_head_attention_8/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2>
<transformer_block_8/multi_head_attention_8/Reshape_3/shape/1?
<transformer_block_8/multi_head_attention_8/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@2>
<transformer_block_8/multi_head_attention_8/Reshape_3/shape/2?
:transformer_block_8/multi_head_attention_8/Reshape_3/shapePackAtransformer_block_8/multi_head_attention_8/strided_slice:output:0Etransformer_block_8/multi_head_attention_8/Reshape_3/shape/1:output:0Etransformer_block_8/multi_head_attention_8/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_8/multi_head_attention_8/Reshape_3/shape?
4transformer_block_8/multi_head_attention_8/Reshape_3Reshape:transformer_block_8/multi_head_attention_8/transpose_3:y:0Ctransformer_block_8/multi_head_attention_8/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :??????????????????@26
4transformer_block_8/multi_head_attention_8/Reshape_3?
Ltransformer_block_8/multi_head_attention_8/dense_99/Tensordot/ReadVariableOpReadVariableOpUtransformer_block_8_multi_head_attention_8_dense_99_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02N
Ltransformer_block_8/multi_head_attention_8/dense_99/Tensordot/ReadVariableOp?
Btransformer_block_8/multi_head_attention_8/dense_99/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2D
Btransformer_block_8/multi_head_attention_8/dense_99/Tensordot/axes?
Btransformer_block_8/multi_head_attention_8/dense_99/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2D
Btransformer_block_8/multi_head_attention_8/dense_99/Tensordot/free?
Ctransformer_block_8/multi_head_attention_8/dense_99/Tensordot/ShapeShape=transformer_block_8/multi_head_attention_8/Reshape_3:output:0*
T0*
_output_shapes
:2E
Ctransformer_block_8/multi_head_attention_8/dense_99/Tensordot/Shape?
Ktransformer_block_8/multi_head_attention_8/dense_99/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Ktransformer_block_8/multi_head_attention_8/dense_99/Tensordot/GatherV2/axis?
Ftransformer_block_8/multi_head_attention_8/dense_99/Tensordot/GatherV2GatherV2Ltransformer_block_8/multi_head_attention_8/dense_99/Tensordot/Shape:output:0Ktransformer_block_8/multi_head_attention_8/dense_99/Tensordot/free:output:0Ttransformer_block_8/multi_head_attention_8/dense_99/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2H
Ftransformer_block_8/multi_head_attention_8/dense_99/Tensordot/GatherV2?
Mtransformer_block_8/multi_head_attention_8/dense_99/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2O
Mtransformer_block_8/multi_head_attention_8/dense_99/Tensordot/GatherV2_1/axis?
Htransformer_block_8/multi_head_attention_8/dense_99/Tensordot/GatherV2_1GatherV2Ltransformer_block_8/multi_head_attention_8/dense_99/Tensordot/Shape:output:0Ktransformer_block_8/multi_head_attention_8/dense_99/Tensordot/axes:output:0Vtransformer_block_8/multi_head_attention_8/dense_99/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2J
Htransformer_block_8/multi_head_attention_8/dense_99/Tensordot/GatherV2_1?
Ctransformer_block_8/multi_head_attention_8/dense_99/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2E
Ctransformer_block_8/multi_head_attention_8/dense_99/Tensordot/Const?
Btransformer_block_8/multi_head_attention_8/dense_99/Tensordot/ProdProdOtransformer_block_8/multi_head_attention_8/dense_99/Tensordot/GatherV2:output:0Ltransformer_block_8/multi_head_attention_8/dense_99/Tensordot/Const:output:0*
T0*
_output_shapes
: 2D
Btransformer_block_8/multi_head_attention_8/dense_99/Tensordot/Prod?
Etransformer_block_8/multi_head_attention_8/dense_99/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2G
Etransformer_block_8/multi_head_attention_8/dense_99/Tensordot/Const_1?
Dtransformer_block_8/multi_head_attention_8/dense_99/Tensordot/Prod_1ProdQtransformer_block_8/multi_head_attention_8/dense_99/Tensordot/GatherV2_1:output:0Ntransformer_block_8/multi_head_attention_8/dense_99/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2F
Dtransformer_block_8/multi_head_attention_8/dense_99/Tensordot/Prod_1?
Itransformer_block_8/multi_head_attention_8/dense_99/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Itransformer_block_8/multi_head_attention_8/dense_99/Tensordot/concat/axis?
Dtransformer_block_8/multi_head_attention_8/dense_99/Tensordot/concatConcatV2Ktransformer_block_8/multi_head_attention_8/dense_99/Tensordot/free:output:0Ktransformer_block_8/multi_head_attention_8/dense_99/Tensordot/axes:output:0Rtransformer_block_8/multi_head_attention_8/dense_99/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2F
Dtransformer_block_8/multi_head_attention_8/dense_99/Tensordot/concat?
Ctransformer_block_8/multi_head_attention_8/dense_99/Tensordot/stackPackKtransformer_block_8/multi_head_attention_8/dense_99/Tensordot/Prod:output:0Mtransformer_block_8/multi_head_attention_8/dense_99/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2E
Ctransformer_block_8/multi_head_attention_8/dense_99/Tensordot/stack?
Gtransformer_block_8/multi_head_attention_8/dense_99/Tensordot/transpose	Transpose=transformer_block_8/multi_head_attention_8/Reshape_3:output:0Mtransformer_block_8/multi_head_attention_8/dense_99/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :??????????????????@2I
Gtransformer_block_8/multi_head_attention_8/dense_99/Tensordot/transpose?
Etransformer_block_8/multi_head_attention_8/dense_99/Tensordot/ReshapeReshapeKtransformer_block_8/multi_head_attention_8/dense_99/Tensordot/transpose:y:0Ltransformer_block_8/multi_head_attention_8/dense_99/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2G
Etransformer_block_8/multi_head_attention_8/dense_99/Tensordot/Reshape?
Dtransformer_block_8/multi_head_attention_8/dense_99/Tensordot/MatMulMatMulNtransformer_block_8/multi_head_attention_8/dense_99/Tensordot/Reshape:output:0Ttransformer_block_8/multi_head_attention_8/dense_99/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2F
Dtransformer_block_8/multi_head_attention_8/dense_99/Tensordot/MatMul?
Etransformer_block_8/multi_head_attention_8/dense_99/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2G
Etransformer_block_8/multi_head_attention_8/dense_99/Tensordot/Const_2?
Ktransformer_block_8/multi_head_attention_8/dense_99/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Ktransformer_block_8/multi_head_attention_8/dense_99/Tensordot/concat_1/axis?
Ftransformer_block_8/multi_head_attention_8/dense_99/Tensordot/concat_1ConcatV2Otransformer_block_8/multi_head_attention_8/dense_99/Tensordot/GatherV2:output:0Ntransformer_block_8/multi_head_attention_8/dense_99/Tensordot/Const_2:output:0Ttransformer_block_8/multi_head_attention_8/dense_99/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2H
Ftransformer_block_8/multi_head_attention_8/dense_99/Tensordot/concat_1?
=transformer_block_8/multi_head_attention_8/dense_99/TensordotReshapeNtransformer_block_8/multi_head_attention_8/dense_99/Tensordot/MatMul:product:0Otransformer_block_8/multi_head_attention_8/dense_99/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :??????????????????@2?
=transformer_block_8/multi_head_attention_8/dense_99/Tensordot?
Jtransformer_block_8/multi_head_attention_8/dense_99/BiasAdd/ReadVariableOpReadVariableOpStransformer_block_8_multi_head_attention_8_dense_99_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02L
Jtransformer_block_8/multi_head_attention_8/dense_99/BiasAdd/ReadVariableOp?
;transformer_block_8/multi_head_attention_8/dense_99/BiasAddBiasAddFtransformer_block_8/multi_head_attention_8/dense_99/Tensordot:output:0Rtransformer_block_8/multi_head_attention_8/dense_99/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????@2=
;transformer_block_8/multi_head_attention_8/dense_99/BiasAdd?
,transformer_block_8/dropout_42/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2.
,transformer_block_8/dropout_42/dropout/Const?
*transformer_block_8/dropout_42/dropout/MulMulDtransformer_block_8/multi_head_attention_8/dense_99/BiasAdd:output:05transformer_block_8/dropout_42/dropout/Const:output:0*
T0*4
_output_shapes"
 :??????????????????@2,
*transformer_block_8/dropout_42/dropout/Mul?
,transformer_block_8/dropout_42/dropout/ShapeShapeDtransformer_block_8/multi_head_attention_8/dense_99/BiasAdd:output:0*
T0*
_output_shapes
:2.
,transformer_block_8/dropout_42/dropout/Shape?
Ctransformer_block_8/dropout_42/dropout/random_uniform/RandomUniformRandomUniform5transformer_block_8/dropout_42/dropout/Shape:output:0*
T0*4
_output_shapes"
 :??????????????????@*
dtype02E
Ctransformer_block_8/dropout_42/dropout/random_uniform/RandomUniform?
5transformer_block_8/dropout_42/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=27
5transformer_block_8/dropout_42/dropout/GreaterEqual/y?
3transformer_block_8/dropout_42/dropout/GreaterEqualGreaterEqualLtransformer_block_8/dropout_42/dropout/random_uniform/RandomUniform:output:0>transformer_block_8/dropout_42/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :??????????????????@25
3transformer_block_8/dropout_42/dropout/GreaterEqual?
+transformer_block_8/dropout_42/dropout/CastCast7transformer_block_8/dropout_42/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :??????????????????@2-
+transformer_block_8/dropout_42/dropout/Cast?
,transformer_block_8/dropout_42/dropout/Mul_1Mul.transformer_block_8/dropout_42/dropout/Mul:z:0/transformer_block_8/dropout_42/dropout/Cast:y:0*
T0*4
_output_shapes"
 :??????????????????@2.
,transformer_block_8/dropout_42/dropout/Mul_1?
transformer_block_8/addAddV2&token_and_position_embedding_9/add:z:00transformer_block_8/dropout_42/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????d@2
transformer_block_8/add?
Itransformer_block_8/layer_normalization_26/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2K
Itransformer_block_8/layer_normalization_26/moments/mean/reduction_indices?
7transformer_block_8/layer_normalization_26/moments/meanMeantransformer_block_8/add:z:0Rtransformer_block_8/layer_normalization_26/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????d*
	keep_dims(29
7transformer_block_8/layer_normalization_26/moments/mean?
?transformer_block_8/layer_normalization_26/moments/StopGradientStopGradient@transformer_block_8/layer_normalization_26/moments/mean:output:0*
T0*+
_output_shapes
:?????????d2A
?transformer_block_8/layer_normalization_26/moments/StopGradient?
Dtransformer_block_8/layer_normalization_26/moments/SquaredDifferenceSquaredDifferencetransformer_block_8/add:z:0Htransformer_block_8/layer_normalization_26/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????d@2F
Dtransformer_block_8/layer_normalization_26/moments/SquaredDifference?
Mtransformer_block_8/layer_normalization_26/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2O
Mtransformer_block_8/layer_normalization_26/moments/variance/reduction_indices?
;transformer_block_8/layer_normalization_26/moments/varianceMeanHtransformer_block_8/layer_normalization_26/moments/SquaredDifference:z:0Vtransformer_block_8/layer_normalization_26/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????d*
	keep_dims(2=
;transformer_block_8/layer_normalization_26/moments/variance?
:transformer_block_8/layer_normalization_26/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52<
:transformer_block_8/layer_normalization_26/batchnorm/add/y?
8transformer_block_8/layer_normalization_26/batchnorm/addAddV2Dtransformer_block_8/layer_normalization_26/moments/variance:output:0Ctransformer_block_8/layer_normalization_26/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????d2:
8transformer_block_8/layer_normalization_26/batchnorm/add?
:transformer_block_8/layer_normalization_26/batchnorm/RsqrtRsqrt<transformer_block_8/layer_normalization_26/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????d2<
:transformer_block_8/layer_normalization_26/batchnorm/Rsqrt?
Gtransformer_block_8/layer_normalization_26/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_8_layer_normalization_26_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02I
Gtransformer_block_8/layer_normalization_26/batchnorm/mul/ReadVariableOp?
8transformer_block_8/layer_normalization_26/batchnorm/mulMul>transformer_block_8/layer_normalization_26/batchnorm/Rsqrt:y:0Otransformer_block_8/layer_normalization_26/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@2:
8transformer_block_8/layer_normalization_26/batchnorm/mul?
:transformer_block_8/layer_normalization_26/batchnorm/mul_1Multransformer_block_8/add:z:0<transformer_block_8/layer_normalization_26/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????d@2<
:transformer_block_8/layer_normalization_26/batchnorm/mul_1?
:transformer_block_8/layer_normalization_26/batchnorm/mul_2Mul@transformer_block_8/layer_normalization_26/moments/mean:output:0<transformer_block_8/layer_normalization_26/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????d@2<
:transformer_block_8/layer_normalization_26/batchnorm/mul_2?
Ctransformer_block_8/layer_normalization_26/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_8_layer_normalization_26_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02E
Ctransformer_block_8/layer_normalization_26/batchnorm/ReadVariableOp?
8transformer_block_8/layer_normalization_26/batchnorm/subSubKtransformer_block_8/layer_normalization_26/batchnorm/ReadVariableOp:value:0>transformer_block_8/layer_normalization_26/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????d@2:
8transformer_block_8/layer_normalization_26/batchnorm/sub?
:transformer_block_8/layer_normalization_26/batchnorm/add_1AddV2>transformer_block_8/layer_normalization_26/batchnorm/mul_1:z:0<transformer_block_8/layer_normalization_26/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????d@2<
:transformer_block_8/layer_normalization_26/batchnorm/add_1?
Ctransformer_block_8/sequential_8/dense_100/Tensordot/ReadVariableOpReadVariableOpLtransformer_block_8_sequential_8_dense_100_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02E
Ctransformer_block_8/sequential_8/dense_100/Tensordot/ReadVariableOp?
9transformer_block_8/sequential_8/dense_100/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2;
9transformer_block_8/sequential_8/dense_100/Tensordot/axes?
9transformer_block_8/sequential_8/dense_100/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2;
9transformer_block_8/sequential_8/dense_100/Tensordot/free?
:transformer_block_8/sequential_8/dense_100/Tensordot/ShapeShape>transformer_block_8/layer_normalization_26/batchnorm/add_1:z:0*
T0*
_output_shapes
:2<
:transformer_block_8/sequential_8/dense_100/Tensordot/Shape?
Btransformer_block_8/sequential_8/dense_100/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block_8/sequential_8/dense_100/Tensordot/GatherV2/axis?
=transformer_block_8/sequential_8/dense_100/Tensordot/GatherV2GatherV2Ctransformer_block_8/sequential_8/dense_100/Tensordot/Shape:output:0Btransformer_block_8/sequential_8/dense_100/Tensordot/free:output:0Ktransformer_block_8/sequential_8/dense_100/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2?
=transformer_block_8/sequential_8/dense_100/Tensordot/GatherV2?
Dtransformer_block_8/sequential_8/dense_100/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2F
Dtransformer_block_8/sequential_8/dense_100/Tensordot/GatherV2_1/axis?
?transformer_block_8/sequential_8/dense_100/Tensordot/GatherV2_1GatherV2Ctransformer_block_8/sequential_8/dense_100/Tensordot/Shape:output:0Btransformer_block_8/sequential_8/dense_100/Tensordot/axes:output:0Mtransformer_block_8/sequential_8/dense_100/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2A
?transformer_block_8/sequential_8/dense_100/Tensordot/GatherV2_1?
:transformer_block_8/sequential_8/dense_100/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block_8/sequential_8/dense_100/Tensordot/Const?
9transformer_block_8/sequential_8/dense_100/Tensordot/ProdProdFtransformer_block_8/sequential_8/dense_100/Tensordot/GatherV2:output:0Ctransformer_block_8/sequential_8/dense_100/Tensordot/Const:output:0*
T0*
_output_shapes
: 2;
9transformer_block_8/sequential_8/dense_100/Tensordot/Prod?
<transformer_block_8/sequential_8/dense_100/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<transformer_block_8/sequential_8/dense_100/Tensordot/Const_1?
;transformer_block_8/sequential_8/dense_100/Tensordot/Prod_1ProdHtransformer_block_8/sequential_8/dense_100/Tensordot/GatherV2_1:output:0Etransformer_block_8/sequential_8/dense_100/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2=
;transformer_block_8/sequential_8/dense_100/Tensordot/Prod_1?
@transformer_block_8/sequential_8/dense_100/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_8/sequential_8/dense_100/Tensordot/concat/axis?
;transformer_block_8/sequential_8/dense_100/Tensordot/concatConcatV2Btransformer_block_8/sequential_8/dense_100/Tensordot/free:output:0Btransformer_block_8/sequential_8/dense_100/Tensordot/axes:output:0Itransformer_block_8/sequential_8/dense_100/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block_8/sequential_8/dense_100/Tensordot/concat?
:transformer_block_8/sequential_8/dense_100/Tensordot/stackPackBtransformer_block_8/sequential_8/dense_100/Tensordot/Prod:output:0Dtransformer_block_8/sequential_8/dense_100/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_8/sequential_8/dense_100/Tensordot/stack?
>transformer_block_8/sequential_8/dense_100/Tensordot/transpose	Transpose>transformer_block_8/layer_normalization_26/batchnorm/add_1:z:0Dtransformer_block_8/sequential_8/dense_100/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????d@2@
>transformer_block_8/sequential_8/dense_100/Tensordot/transpose?
<transformer_block_8/sequential_8/dense_100/Tensordot/ReshapeReshapeBtransformer_block_8/sequential_8/dense_100/Tensordot/transpose:y:0Ctransformer_block_8/sequential_8/dense_100/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2>
<transformer_block_8/sequential_8/dense_100/Tensordot/Reshape?
;transformer_block_8/sequential_8/dense_100/Tensordot/MatMulMatMulEtransformer_block_8/sequential_8/dense_100/Tensordot/Reshape:output:0Ktransformer_block_8/sequential_8/dense_100/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2=
;transformer_block_8/sequential_8/dense_100/Tensordot/MatMul?
<transformer_block_8/sequential_8/dense_100/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2>
<transformer_block_8/sequential_8/dense_100/Tensordot/Const_2?
Btransformer_block_8/sequential_8/dense_100/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block_8/sequential_8/dense_100/Tensordot/concat_1/axis?
=transformer_block_8/sequential_8/dense_100/Tensordot/concat_1ConcatV2Ftransformer_block_8/sequential_8/dense_100/Tensordot/GatherV2:output:0Etransformer_block_8/sequential_8/dense_100/Tensordot/Const_2:output:0Ktransformer_block_8/sequential_8/dense_100/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2?
=transformer_block_8/sequential_8/dense_100/Tensordot/concat_1?
4transformer_block_8/sequential_8/dense_100/TensordotReshapeEtransformer_block_8/sequential_8/dense_100/Tensordot/MatMul:product:0Ftransformer_block_8/sequential_8/dense_100/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????d 26
4transformer_block_8/sequential_8/dense_100/Tensordot?
Atransformer_block_8/sequential_8/dense_100/BiasAdd/ReadVariableOpReadVariableOpJtransformer_block_8_sequential_8_dense_100_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02C
Atransformer_block_8/sequential_8/dense_100/BiasAdd/ReadVariableOp?
2transformer_block_8/sequential_8/dense_100/BiasAddBiasAdd=transformer_block_8/sequential_8/dense_100/Tensordot:output:0Itransformer_block_8/sequential_8/dense_100/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d 24
2transformer_block_8/sequential_8/dense_100/BiasAdd?
/transformer_block_8/sequential_8/dense_100/ReluRelu;transformer_block_8/sequential_8/dense_100/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d 21
/transformer_block_8/sequential_8/dense_100/Relu?
Ctransformer_block_8/sequential_8/dense_101/Tensordot/ReadVariableOpReadVariableOpLtransformer_block_8_sequential_8_dense_101_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02E
Ctransformer_block_8/sequential_8/dense_101/Tensordot/ReadVariableOp?
9transformer_block_8/sequential_8/dense_101/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2;
9transformer_block_8/sequential_8/dense_101/Tensordot/axes?
9transformer_block_8/sequential_8/dense_101/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2;
9transformer_block_8/sequential_8/dense_101/Tensordot/free?
:transformer_block_8/sequential_8/dense_101/Tensordot/ShapeShape=transformer_block_8/sequential_8/dense_100/Relu:activations:0*
T0*
_output_shapes
:2<
:transformer_block_8/sequential_8/dense_101/Tensordot/Shape?
Btransformer_block_8/sequential_8/dense_101/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block_8/sequential_8/dense_101/Tensordot/GatherV2/axis?
=transformer_block_8/sequential_8/dense_101/Tensordot/GatherV2GatherV2Ctransformer_block_8/sequential_8/dense_101/Tensordot/Shape:output:0Btransformer_block_8/sequential_8/dense_101/Tensordot/free:output:0Ktransformer_block_8/sequential_8/dense_101/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2?
=transformer_block_8/sequential_8/dense_101/Tensordot/GatherV2?
Dtransformer_block_8/sequential_8/dense_101/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2F
Dtransformer_block_8/sequential_8/dense_101/Tensordot/GatherV2_1/axis?
?transformer_block_8/sequential_8/dense_101/Tensordot/GatherV2_1GatherV2Ctransformer_block_8/sequential_8/dense_101/Tensordot/Shape:output:0Btransformer_block_8/sequential_8/dense_101/Tensordot/axes:output:0Mtransformer_block_8/sequential_8/dense_101/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2A
?transformer_block_8/sequential_8/dense_101/Tensordot/GatherV2_1?
:transformer_block_8/sequential_8/dense_101/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block_8/sequential_8/dense_101/Tensordot/Const?
9transformer_block_8/sequential_8/dense_101/Tensordot/ProdProdFtransformer_block_8/sequential_8/dense_101/Tensordot/GatherV2:output:0Ctransformer_block_8/sequential_8/dense_101/Tensordot/Const:output:0*
T0*
_output_shapes
: 2;
9transformer_block_8/sequential_8/dense_101/Tensordot/Prod?
<transformer_block_8/sequential_8/dense_101/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<transformer_block_8/sequential_8/dense_101/Tensordot/Const_1?
;transformer_block_8/sequential_8/dense_101/Tensordot/Prod_1ProdHtransformer_block_8/sequential_8/dense_101/Tensordot/GatherV2_1:output:0Etransformer_block_8/sequential_8/dense_101/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2=
;transformer_block_8/sequential_8/dense_101/Tensordot/Prod_1?
@transformer_block_8/sequential_8/dense_101/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_8/sequential_8/dense_101/Tensordot/concat/axis?
;transformer_block_8/sequential_8/dense_101/Tensordot/concatConcatV2Btransformer_block_8/sequential_8/dense_101/Tensordot/free:output:0Btransformer_block_8/sequential_8/dense_101/Tensordot/axes:output:0Itransformer_block_8/sequential_8/dense_101/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block_8/sequential_8/dense_101/Tensordot/concat?
:transformer_block_8/sequential_8/dense_101/Tensordot/stackPackBtransformer_block_8/sequential_8/dense_101/Tensordot/Prod:output:0Dtransformer_block_8/sequential_8/dense_101/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_8/sequential_8/dense_101/Tensordot/stack?
>transformer_block_8/sequential_8/dense_101/Tensordot/transpose	Transpose=transformer_block_8/sequential_8/dense_100/Relu:activations:0Dtransformer_block_8/sequential_8/dense_101/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????d 2@
>transformer_block_8/sequential_8/dense_101/Tensordot/transpose?
<transformer_block_8/sequential_8/dense_101/Tensordot/ReshapeReshapeBtransformer_block_8/sequential_8/dense_101/Tensordot/transpose:y:0Ctransformer_block_8/sequential_8/dense_101/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2>
<transformer_block_8/sequential_8/dense_101/Tensordot/Reshape?
;transformer_block_8/sequential_8/dense_101/Tensordot/MatMulMatMulEtransformer_block_8/sequential_8/dense_101/Tensordot/Reshape:output:0Ktransformer_block_8/sequential_8/dense_101/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2=
;transformer_block_8/sequential_8/dense_101/Tensordot/MatMul?
<transformer_block_8/sequential_8/dense_101/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2>
<transformer_block_8/sequential_8/dense_101/Tensordot/Const_2?
Btransformer_block_8/sequential_8/dense_101/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block_8/sequential_8/dense_101/Tensordot/concat_1/axis?
=transformer_block_8/sequential_8/dense_101/Tensordot/concat_1ConcatV2Ftransformer_block_8/sequential_8/dense_101/Tensordot/GatherV2:output:0Etransformer_block_8/sequential_8/dense_101/Tensordot/Const_2:output:0Ktransformer_block_8/sequential_8/dense_101/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2?
=transformer_block_8/sequential_8/dense_101/Tensordot/concat_1?
4transformer_block_8/sequential_8/dense_101/TensordotReshapeEtransformer_block_8/sequential_8/dense_101/Tensordot/MatMul:product:0Ftransformer_block_8/sequential_8/dense_101/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????d@26
4transformer_block_8/sequential_8/dense_101/Tensordot?
Atransformer_block_8/sequential_8/dense_101/BiasAdd/ReadVariableOpReadVariableOpJtransformer_block_8_sequential_8_dense_101_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02C
Atransformer_block_8/sequential_8/dense_101/BiasAdd/ReadVariableOp?
2transformer_block_8/sequential_8/dense_101/BiasAddBiasAdd=transformer_block_8/sequential_8/dense_101/Tensordot:output:0Itransformer_block_8/sequential_8/dense_101/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@24
2transformer_block_8/sequential_8/dense_101/BiasAdd?
,transformer_block_8/dropout_43/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2.
,transformer_block_8/dropout_43/dropout/Const?
*transformer_block_8/dropout_43/dropout/MulMul;transformer_block_8/sequential_8/dense_101/BiasAdd:output:05transformer_block_8/dropout_43/dropout/Const:output:0*
T0*+
_output_shapes
:?????????d@2,
*transformer_block_8/dropout_43/dropout/Mul?
,transformer_block_8/dropout_43/dropout/ShapeShape;transformer_block_8/sequential_8/dense_101/BiasAdd:output:0*
T0*
_output_shapes
:2.
,transformer_block_8/dropout_43/dropout/Shape?
Ctransformer_block_8/dropout_43/dropout/random_uniform/RandomUniformRandomUniform5transformer_block_8/dropout_43/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????d@*
dtype02E
Ctransformer_block_8/dropout_43/dropout/random_uniform/RandomUniform?
5transformer_block_8/dropout_43/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=27
5transformer_block_8/dropout_43/dropout/GreaterEqual/y?
3transformer_block_8/dropout_43/dropout/GreaterEqualGreaterEqualLtransformer_block_8/dropout_43/dropout/random_uniform/RandomUniform:output:0>transformer_block_8/dropout_43/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????d@25
3transformer_block_8/dropout_43/dropout/GreaterEqual?
+transformer_block_8/dropout_43/dropout/CastCast7transformer_block_8/dropout_43/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????d@2-
+transformer_block_8/dropout_43/dropout/Cast?
,transformer_block_8/dropout_43/dropout/Mul_1Mul.transformer_block_8/dropout_43/dropout/Mul:z:0/transformer_block_8/dropout_43/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????d@2.
,transformer_block_8/dropout_43/dropout/Mul_1?
transformer_block_8/add_1AddV2>transformer_block_8/layer_normalization_26/batchnorm/add_1:z:00transformer_block_8/dropout_43/dropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????d@2
transformer_block_8/add_1?
Itransformer_block_8/layer_normalization_27/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2K
Itransformer_block_8/layer_normalization_27/moments/mean/reduction_indices?
7transformer_block_8/layer_normalization_27/moments/meanMeantransformer_block_8/add_1:z:0Rtransformer_block_8/layer_normalization_27/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????d*
	keep_dims(29
7transformer_block_8/layer_normalization_27/moments/mean?
?transformer_block_8/layer_normalization_27/moments/StopGradientStopGradient@transformer_block_8/layer_normalization_27/moments/mean:output:0*
T0*+
_output_shapes
:?????????d2A
?transformer_block_8/layer_normalization_27/moments/StopGradient?
Dtransformer_block_8/layer_normalization_27/moments/SquaredDifferenceSquaredDifferencetransformer_block_8/add_1:z:0Htransformer_block_8/layer_normalization_27/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????d@2F
Dtransformer_block_8/layer_normalization_27/moments/SquaredDifference?
Mtransformer_block_8/layer_normalization_27/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2O
Mtransformer_block_8/layer_normalization_27/moments/variance/reduction_indices?
;transformer_block_8/layer_normalization_27/moments/varianceMeanHtransformer_block_8/layer_normalization_27/moments/SquaredDifference:z:0Vtransformer_block_8/layer_normalization_27/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????d*
	keep_dims(2=
;transformer_block_8/layer_normalization_27/moments/variance?
:transformer_block_8/layer_normalization_27/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52<
:transformer_block_8/layer_normalization_27/batchnorm/add/y?
8transformer_block_8/layer_normalization_27/batchnorm/addAddV2Dtransformer_block_8/layer_normalization_27/moments/variance:output:0Ctransformer_block_8/layer_normalization_27/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????d2:
8transformer_block_8/layer_normalization_27/batchnorm/add?
:transformer_block_8/layer_normalization_27/batchnorm/RsqrtRsqrt<transformer_block_8/layer_normalization_27/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????d2<
:transformer_block_8/layer_normalization_27/batchnorm/Rsqrt?
Gtransformer_block_8/layer_normalization_27/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_8_layer_normalization_27_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02I
Gtransformer_block_8/layer_normalization_27/batchnorm/mul/ReadVariableOp?
8transformer_block_8/layer_normalization_27/batchnorm/mulMul>transformer_block_8/layer_normalization_27/batchnorm/Rsqrt:y:0Otransformer_block_8/layer_normalization_27/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@2:
8transformer_block_8/layer_normalization_27/batchnorm/mul?
:transformer_block_8/layer_normalization_27/batchnorm/mul_1Multransformer_block_8/add_1:z:0<transformer_block_8/layer_normalization_27/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????d@2<
:transformer_block_8/layer_normalization_27/batchnorm/mul_1?
:transformer_block_8/layer_normalization_27/batchnorm/mul_2Mul@transformer_block_8/layer_normalization_27/moments/mean:output:0<transformer_block_8/layer_normalization_27/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????d@2<
:transformer_block_8/layer_normalization_27/batchnorm/mul_2?
Ctransformer_block_8/layer_normalization_27/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_8_layer_normalization_27_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02E
Ctransformer_block_8/layer_normalization_27/batchnorm/ReadVariableOp?
8transformer_block_8/layer_normalization_27/batchnorm/subSubKtransformer_block_8/layer_normalization_27/batchnorm/ReadVariableOp:value:0>transformer_block_8/layer_normalization_27/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????d@2:
8transformer_block_8/layer_normalization_27/batchnorm/sub?
:transformer_block_8/layer_normalization_27/batchnorm/add_1AddV2>transformer_block_8/layer_normalization_27/batchnorm/mul_1:z:0<transformer_block_8/layer_normalization_27/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????d@2<
:transformer_block_8/layer_normalization_27/batchnorm/add_1?
1global_average_pooling1d_8/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :23
1global_average_pooling1d_8/Mean/reduction_indices?
global_average_pooling1d_8/MeanMean>transformer_block_8/layer_normalization_27/batchnorm/add_1:z:0:global_average_pooling1d_8/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@2!
global_average_pooling1d_8/Meany
dropout_44/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_44/dropout/Const?
dropout_44/dropout/MulMul(global_average_pooling1d_8/Mean:output:0!dropout_44/dropout/Const:output:0*
T0*'
_output_shapes
:?????????@2
dropout_44/dropout/Mul?
dropout_44/dropout/ShapeShape(global_average_pooling1d_8/Mean:output:0*
T0*
_output_shapes
:2
dropout_44/dropout/Shape?
/dropout_44/dropout/random_uniform/RandomUniformRandomUniform!dropout_44/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype021
/dropout_44/dropout/random_uniform/RandomUniform?
!dropout_44/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2#
!dropout_44/dropout/GreaterEqual/y?
dropout_44/dropout/GreaterEqualGreaterEqual8dropout_44/dropout/random_uniform/RandomUniform:output:0*dropout_44/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2!
dropout_44/dropout/GreaterEqual?
dropout_44/dropout/CastCast#dropout_44/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
dropout_44/dropout/Cast?
dropout_44/dropout/Mul_1Muldropout_44/dropout/Mul:z:0dropout_44/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@2
dropout_44/dropout/Mul_1?
dense_102/MatMul/ReadVariableOpReadVariableOp(dense_102_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02!
dense_102/MatMul/ReadVariableOp?
dense_102/MatMulMatMuldropout_44/dropout/Mul_1:z:0'dense_102/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_102/MatMul?
 dense_102/BiasAdd/ReadVariableOpReadVariableOp)dense_102_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_102/BiasAdd/ReadVariableOp?
dense_102/BiasAddBiasAdddense_102/MatMul:product:0(dense_102/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_102/BiasAddv
dense_102/ReluReludense_102/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_102/Reluy
dropout_45/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_45/dropout/Const?
dropout_45/dropout/MulMuldense_102/Relu:activations:0!dropout_45/dropout/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout_45/dropout/Mul?
dropout_45/dropout/ShapeShapedense_102/Relu:activations:0*
T0*
_output_shapes
:2
dropout_45/dropout/Shape?
/dropout_45/dropout/random_uniform/RandomUniformRandomUniform!dropout_45/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype021
/dropout_45/dropout/random_uniform/RandomUniform?
!dropout_45/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2#
!dropout_45/dropout/GreaterEqual/y?
dropout_45/dropout/GreaterEqualGreaterEqual8dropout_45/dropout/random_uniform/RandomUniform:output:0*dropout_45/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2!
dropout_45/dropout/GreaterEqual?
dropout_45/dropout/CastCast#dropout_45/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout_45/dropout/Cast?
dropout_45/dropout/Mul_1Muldropout_45/dropout/Mul:z:0dropout_45/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout_45/dropout/Mul_1?
dense_103/MatMul/ReadVariableOpReadVariableOp(dense_103_matmul_readvariableop_resource*
_output_shapes

:*
dtype02!
dense_103/MatMul/ReadVariableOp?
dense_103/MatMulMatMuldropout_45/dropout/Mul_1:z:0'dense_103/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_103/MatMul?
 dense_103/BiasAdd/ReadVariableOpReadVariableOp)dense_103_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_103/BiasAdd/ReadVariableOp?
dense_103/BiasAddBiasAdddense_103/MatMul:product:0(dense_103/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_103/BiasAdd
dense_103/SoftmaxSoftmaxdense_103/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_103/Softmax?
IdentityIdentitydense_103/Softmax:softmax:0!^dense_102/BiasAdd/ReadVariableOp ^dense_102/MatMul/ReadVariableOp!^dense_103/BiasAdd/ReadVariableOp ^dense_103/MatMul/ReadVariableOp=^token_and_position_embedding_9/embedding_20/embedding_lookup=^token_and_position_embedding_9/embedding_21/embedding_lookupD^transformer_block_8/layer_normalization_26/batchnorm/ReadVariableOpH^transformer_block_8/layer_normalization_26/batchnorm/mul/ReadVariableOpD^transformer_block_8/layer_normalization_27/batchnorm/ReadVariableOpH^transformer_block_8/layer_normalization_27/batchnorm/mul/ReadVariableOpK^transformer_block_8/multi_head_attention_8/dense_96/BiasAdd/ReadVariableOpM^transformer_block_8/multi_head_attention_8/dense_96/Tensordot/ReadVariableOpK^transformer_block_8/multi_head_attention_8/dense_97/BiasAdd/ReadVariableOpM^transformer_block_8/multi_head_attention_8/dense_97/Tensordot/ReadVariableOpK^transformer_block_8/multi_head_attention_8/dense_98/BiasAdd/ReadVariableOpM^transformer_block_8/multi_head_attention_8/dense_98/Tensordot/ReadVariableOpK^transformer_block_8/multi_head_attention_8/dense_99/BiasAdd/ReadVariableOpM^transformer_block_8/multi_head_attention_8/dense_99/Tensordot/ReadVariableOpB^transformer_block_8/sequential_8/dense_100/BiasAdd/ReadVariableOpD^transformer_block_8/sequential_8/dense_100/Tensordot/ReadVariableOpB^transformer_block_8/sequential_8/dense_101/BiasAdd/ReadVariableOpD^transformer_block_8/sequential_8/dense_101/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????d: : : : : : : : : : : : : : : : : : : : : : 2D
 dense_102/BiasAdd/ReadVariableOp dense_102/BiasAdd/ReadVariableOp2B
dense_102/MatMul/ReadVariableOpdense_102/MatMul/ReadVariableOp2D
 dense_103/BiasAdd/ReadVariableOp dense_103/BiasAdd/ReadVariableOp2B
dense_103/MatMul/ReadVariableOpdense_103/MatMul/ReadVariableOp2|
<token_and_position_embedding_9/embedding_20/embedding_lookup<token_and_position_embedding_9/embedding_20/embedding_lookup2|
<token_and_position_embedding_9/embedding_21/embedding_lookup<token_and_position_embedding_9/embedding_21/embedding_lookup2?
Ctransformer_block_8/layer_normalization_26/batchnorm/ReadVariableOpCtransformer_block_8/layer_normalization_26/batchnorm/ReadVariableOp2?
Gtransformer_block_8/layer_normalization_26/batchnorm/mul/ReadVariableOpGtransformer_block_8/layer_normalization_26/batchnorm/mul/ReadVariableOp2?
Ctransformer_block_8/layer_normalization_27/batchnorm/ReadVariableOpCtransformer_block_8/layer_normalization_27/batchnorm/ReadVariableOp2?
Gtransformer_block_8/layer_normalization_27/batchnorm/mul/ReadVariableOpGtransformer_block_8/layer_normalization_27/batchnorm/mul/ReadVariableOp2?
Jtransformer_block_8/multi_head_attention_8/dense_96/BiasAdd/ReadVariableOpJtransformer_block_8/multi_head_attention_8/dense_96/BiasAdd/ReadVariableOp2?
Ltransformer_block_8/multi_head_attention_8/dense_96/Tensordot/ReadVariableOpLtransformer_block_8/multi_head_attention_8/dense_96/Tensordot/ReadVariableOp2?
Jtransformer_block_8/multi_head_attention_8/dense_97/BiasAdd/ReadVariableOpJtransformer_block_8/multi_head_attention_8/dense_97/BiasAdd/ReadVariableOp2?
Ltransformer_block_8/multi_head_attention_8/dense_97/Tensordot/ReadVariableOpLtransformer_block_8/multi_head_attention_8/dense_97/Tensordot/ReadVariableOp2?
Jtransformer_block_8/multi_head_attention_8/dense_98/BiasAdd/ReadVariableOpJtransformer_block_8/multi_head_attention_8/dense_98/BiasAdd/ReadVariableOp2?
Ltransformer_block_8/multi_head_attention_8/dense_98/Tensordot/ReadVariableOpLtransformer_block_8/multi_head_attention_8/dense_98/Tensordot/ReadVariableOp2?
Jtransformer_block_8/multi_head_attention_8/dense_99/BiasAdd/ReadVariableOpJtransformer_block_8/multi_head_attention_8/dense_99/BiasAdd/ReadVariableOp2?
Ltransformer_block_8/multi_head_attention_8/dense_99/Tensordot/ReadVariableOpLtransformer_block_8/multi_head_attention_8/dense_99/Tensordot/ReadVariableOp2?
Atransformer_block_8/sequential_8/dense_100/BiasAdd/ReadVariableOpAtransformer_block_8/sequential_8/dense_100/BiasAdd/ReadVariableOp2?
Ctransformer_block_8/sequential_8/dense_100/Tensordot/ReadVariableOpCtransformer_block_8/sequential_8/dense_100/Tensordot/ReadVariableOp2?
Atransformer_block_8/sequential_8/dense_101/BiasAdd/ReadVariableOpAtransformer_block_8/sequential_8/dense_101/BiasAdd/ReadVariableOp2?
Ctransformer_block_8/sequential_8/dense_101/Tensordot/ReadVariableOpCtransformer_block_8/sequential_8/dense_101/Tensordot/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?

?
D__inference_dense_103_layer_call_and_return_conditional_losses_91392

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
3__inference_transformer_block_8_layer_call_fn_93026

inputs
unknown:@@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@ 

unknown_10: 

unknown_11: @

unknown_12:@

unknown_13:@

unknown_14:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d@*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_transformer_block_8_layer_call_and_return_conditional_losses_918152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????d@: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????d@
 
_user_specified_nameinputs
?
?
>__inference_token_and_position_embedding_9_layer_call_fn_92928
x
unknown:d@
	unknown_0:	?@@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *b
f]R[
Y__inference_token_and_position_embedding_9_layer_call_and_return_conditional_losses_910592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:?????????d

_user_specified_namex
?/
?	
B__inference_model_8_layer_call_and_return_conditional_losses_91966

inputs6
$token_and_position_embedding_9_91914:d@7
$token_and_position_embedding_9_91916:	?@@+
transformer_block_8_91919:@@'
transformer_block_8_91921:@+
transformer_block_8_91923:@@'
transformer_block_8_91925:@+
transformer_block_8_91927:@@'
transformer_block_8_91929:@+
transformer_block_8_91931:@@'
transformer_block_8_91933:@'
transformer_block_8_91935:@'
transformer_block_8_91937:@+
transformer_block_8_91939:@ '
transformer_block_8_91941: +
transformer_block_8_91943: @'
transformer_block_8_91945:@'
transformer_block_8_91947:@'
transformer_block_8_91949:@!
dense_102_91954:@
dense_102_91956:!
dense_103_91960:
dense_103_91962:
identity??!dense_102/StatefulPartitionedCall?!dense_103/StatefulPartitionedCall?"dropout_44/StatefulPartitionedCall?"dropout_45/StatefulPartitionedCall?6token_and_position_embedding_9/StatefulPartitionedCall?+transformer_block_8/StatefulPartitionedCall?
6token_and_position_embedding_9/StatefulPartitionedCallStatefulPartitionedCallinputs$token_and_position_embedding_9_91914$token_and_position_embedding_9_91916*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *b
f]R[
Y__inference_token_and_position_embedding_9_layer_call_and_return_conditional_losses_9105928
6token_and_position_embedding_9/StatefulPartitionedCall?
+transformer_block_8/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_9/StatefulPartitionedCall:output:0transformer_block_8_91919transformer_block_8_91921transformer_block_8_91923transformer_block_8_91925transformer_block_8_91927transformer_block_8_91929transformer_block_8_91931transformer_block_8_91933transformer_block_8_91935transformer_block_8_91937transformer_block_8_91939transformer_block_8_91941transformer_block_8_91943transformer_block_8_91945transformer_block_8_91947transformer_block_8_91949*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d@*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_transformer_block_8_layer_call_and_return_conditional_losses_918152-
+transformer_block_8/StatefulPartitionedCall?
*global_average_pooling1d_8/PartitionedCallPartitionedCall4transformer_block_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_913482,
*global_average_pooling1d_8/PartitionedCall?
"dropout_44/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_44_layer_call_and_return_conditional_losses_915092$
"dropout_44/StatefulPartitionedCall?
!dense_102/StatefulPartitionedCallStatefulPartitionedCall+dropout_44/StatefulPartitionedCall:output:0dense_102_91954dense_102_91956*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_102_layer_call_and_return_conditional_losses_913682#
!dense_102/StatefulPartitionedCall?
"dropout_45/StatefulPartitionedCallStatefulPartitionedCall*dense_102/StatefulPartitionedCall:output:0#^dropout_44/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_45_layer_call_and_return_conditional_losses_914762$
"dropout_45/StatefulPartitionedCall?
!dense_103/StatefulPartitionedCallStatefulPartitionedCall+dropout_45/StatefulPartitionedCall:output:0dense_103_91960dense_103_91962*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_103_layer_call_and_return_conditional_losses_913922#
!dense_103/StatefulPartitionedCall?
IdentityIdentity*dense_103/StatefulPartitionedCall:output:0"^dense_102/StatefulPartitionedCall"^dense_103/StatefulPartitionedCall#^dropout_44/StatefulPartitionedCall#^dropout_45/StatefulPartitionedCall7^token_and_position_embedding_9/StatefulPartitionedCall,^transformer_block_8/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????d: : : : : : : : : : : : : : : : : : : : : : 2F
!dense_102/StatefulPartitionedCall!dense_102/StatefulPartitionedCall2F
!dense_103/StatefulPartitionedCall!dense_103/StatefulPartitionedCall2H
"dropout_44/StatefulPartitionedCall"dropout_44/StatefulPartitionedCall2H
"dropout_45/StatefulPartitionedCall"dropout_45/StatefulPartitionedCall2p
6token_and_position_embedding_9/StatefulPartitionedCall6token_and_position_embedding_9/StatefulPartitionedCall2Z
+transformer_block_8/StatefulPartitionedCall+transformer_block_8/StatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
F
*__inference_dropout_45_layer_call_fn_93602

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_45_layer_call_and_return_conditional_losses_913792
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
B__inference_model_8_layer_call_and_return_conditional_losses_92609

inputsT
Btoken_and_position_embedding_9_embedding_21_embedding_lookup_92338:d@U
Btoken_and_position_embedding_9_embedding_20_embedding_lookup_92344:	?@@g
Utransformer_block_8_multi_head_attention_8_dense_96_tensordot_readvariableop_resource:@@a
Stransformer_block_8_multi_head_attention_8_dense_96_biasadd_readvariableop_resource:@g
Utransformer_block_8_multi_head_attention_8_dense_97_tensordot_readvariableop_resource:@@a
Stransformer_block_8_multi_head_attention_8_dense_97_biasadd_readvariableop_resource:@g
Utransformer_block_8_multi_head_attention_8_dense_98_tensordot_readvariableop_resource:@@a
Stransformer_block_8_multi_head_attention_8_dense_98_biasadd_readvariableop_resource:@g
Utransformer_block_8_multi_head_attention_8_dense_99_tensordot_readvariableop_resource:@@a
Stransformer_block_8_multi_head_attention_8_dense_99_biasadd_readvariableop_resource:@^
Ptransformer_block_8_layer_normalization_26_batchnorm_mul_readvariableop_resource:@Z
Ltransformer_block_8_layer_normalization_26_batchnorm_readvariableop_resource:@^
Ltransformer_block_8_sequential_8_dense_100_tensordot_readvariableop_resource:@ X
Jtransformer_block_8_sequential_8_dense_100_biasadd_readvariableop_resource: ^
Ltransformer_block_8_sequential_8_dense_101_tensordot_readvariableop_resource: @X
Jtransformer_block_8_sequential_8_dense_101_biasadd_readvariableop_resource:@^
Ptransformer_block_8_layer_normalization_27_batchnorm_mul_readvariableop_resource:@Z
Ltransformer_block_8_layer_normalization_27_batchnorm_readvariableop_resource:@:
(dense_102_matmul_readvariableop_resource:@7
)dense_102_biasadd_readvariableop_resource::
(dense_103_matmul_readvariableop_resource:7
)dense_103_biasadd_readvariableop_resource:
identity?? dense_102/BiasAdd/ReadVariableOp?dense_102/MatMul/ReadVariableOp? dense_103/BiasAdd/ReadVariableOp?dense_103/MatMul/ReadVariableOp?<token_and_position_embedding_9/embedding_20/embedding_lookup?<token_and_position_embedding_9/embedding_21/embedding_lookup?Ctransformer_block_8/layer_normalization_26/batchnorm/ReadVariableOp?Gtransformer_block_8/layer_normalization_26/batchnorm/mul/ReadVariableOp?Ctransformer_block_8/layer_normalization_27/batchnorm/ReadVariableOp?Gtransformer_block_8/layer_normalization_27/batchnorm/mul/ReadVariableOp?Jtransformer_block_8/multi_head_attention_8/dense_96/BiasAdd/ReadVariableOp?Ltransformer_block_8/multi_head_attention_8/dense_96/Tensordot/ReadVariableOp?Jtransformer_block_8/multi_head_attention_8/dense_97/BiasAdd/ReadVariableOp?Ltransformer_block_8/multi_head_attention_8/dense_97/Tensordot/ReadVariableOp?Jtransformer_block_8/multi_head_attention_8/dense_98/BiasAdd/ReadVariableOp?Ltransformer_block_8/multi_head_attention_8/dense_98/Tensordot/ReadVariableOp?Jtransformer_block_8/multi_head_attention_8/dense_99/BiasAdd/ReadVariableOp?Ltransformer_block_8/multi_head_attention_8/dense_99/Tensordot/ReadVariableOp?Atransformer_block_8/sequential_8/dense_100/BiasAdd/ReadVariableOp?Ctransformer_block_8/sequential_8/dense_100/Tensordot/ReadVariableOp?Atransformer_block_8/sequential_8/dense_101/BiasAdd/ReadVariableOp?Ctransformer_block_8/sequential_8/dense_101/Tensordot/ReadVariableOp?
$token_and_position_embedding_9/ShapeShapeinputs*
T0*
_output_shapes
:2&
$token_and_position_embedding_9/Shape?
2token_and_position_embedding_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????24
2token_and_position_embedding_9/strided_slice/stack?
4token_and_position_embedding_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 26
4token_and_position_embedding_9/strided_slice/stack_1?
4token_and_position_embedding_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4token_and_position_embedding_9/strided_slice/stack_2?
,token_and_position_embedding_9/strided_sliceStridedSlice-token_and_position_embedding_9/Shape:output:0;token_and_position_embedding_9/strided_slice/stack:output:0=token_and_position_embedding_9/strided_slice/stack_1:output:0=token_and_position_embedding_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,token_and_position_embedding_9/strided_slice?
*token_and_position_embedding_9/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2,
*token_and_position_embedding_9/range/start?
*token_and_position_embedding_9/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2,
*token_and_position_embedding_9/range/delta?
$token_and_position_embedding_9/rangeRange3token_and_position_embedding_9/range/start:output:05token_and_position_embedding_9/strided_slice:output:03token_and_position_embedding_9/range/delta:output:0*#
_output_shapes
:?????????2&
$token_and_position_embedding_9/range?
<token_and_position_embedding_9/embedding_21/embedding_lookupResourceGatherBtoken_and_position_embedding_9_embedding_21_embedding_lookup_92338-token_and_position_embedding_9/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*U
_classK
IGloc:@token_and_position_embedding_9/embedding_21/embedding_lookup/92338*'
_output_shapes
:?????????@*
dtype02>
<token_and_position_embedding_9/embedding_21/embedding_lookup?
Etoken_and_position_embedding_9/embedding_21/embedding_lookup/IdentityIdentityEtoken_and_position_embedding_9/embedding_21/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@token_and_position_embedding_9/embedding_21/embedding_lookup/92338*'
_output_shapes
:?????????@2G
Etoken_and_position_embedding_9/embedding_21/embedding_lookup/Identity?
Gtoken_and_position_embedding_9/embedding_21/embedding_lookup/Identity_1IdentityNtoken_and_position_embedding_9/embedding_21/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????@2I
Gtoken_and_position_embedding_9/embedding_21/embedding_lookup/Identity_1?
0token_and_position_embedding_9/embedding_20/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????d22
0token_and_position_embedding_9/embedding_20/Cast?
<token_and_position_embedding_9/embedding_20/embedding_lookupResourceGatherBtoken_and_position_embedding_9_embedding_20_embedding_lookup_923444token_and_position_embedding_9/embedding_20/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*U
_classK
IGloc:@token_and_position_embedding_9/embedding_20/embedding_lookup/92344*+
_output_shapes
:?????????d@*
dtype02>
<token_and_position_embedding_9/embedding_20/embedding_lookup?
Etoken_and_position_embedding_9/embedding_20/embedding_lookup/IdentityIdentityEtoken_and_position_embedding_9/embedding_20/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@token_and_position_embedding_9/embedding_20/embedding_lookup/92344*+
_output_shapes
:?????????d@2G
Etoken_and_position_embedding_9/embedding_20/embedding_lookup/Identity?
Gtoken_and_position_embedding_9/embedding_20/embedding_lookup/Identity_1IdentityNtoken_and_position_embedding_9/embedding_20/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????d@2I
Gtoken_and_position_embedding_9/embedding_20/embedding_lookup/Identity_1?
"token_and_position_embedding_9/addAddV2Ptoken_and_position_embedding_9/embedding_20/embedding_lookup/Identity_1:output:0Ptoken_and_position_embedding_9/embedding_21/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:?????????d@2$
"token_and_position_embedding_9/add?
0transformer_block_8/multi_head_attention_8/ShapeShape&token_and_position_embedding_9/add:z:0*
T0*
_output_shapes
:22
0transformer_block_8/multi_head_attention_8/Shape?
>transformer_block_8/multi_head_attention_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>transformer_block_8/multi_head_attention_8/strided_slice/stack?
@transformer_block_8/multi_head_attention_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@transformer_block_8/multi_head_attention_8/strided_slice/stack_1?
@transformer_block_8/multi_head_attention_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@transformer_block_8/multi_head_attention_8/strided_slice/stack_2?
8transformer_block_8/multi_head_attention_8/strided_sliceStridedSlice9transformer_block_8/multi_head_attention_8/Shape:output:0Gtransformer_block_8/multi_head_attention_8/strided_slice/stack:output:0Itransformer_block_8/multi_head_attention_8/strided_slice/stack_1:output:0Itransformer_block_8/multi_head_attention_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2:
8transformer_block_8/multi_head_attention_8/strided_slice?
Ltransformer_block_8/multi_head_attention_8/dense_96/Tensordot/ReadVariableOpReadVariableOpUtransformer_block_8_multi_head_attention_8_dense_96_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02N
Ltransformer_block_8/multi_head_attention_8/dense_96/Tensordot/ReadVariableOp?
Btransformer_block_8/multi_head_attention_8/dense_96/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2D
Btransformer_block_8/multi_head_attention_8/dense_96/Tensordot/axes?
Btransformer_block_8/multi_head_attention_8/dense_96/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2D
Btransformer_block_8/multi_head_attention_8/dense_96/Tensordot/free?
Ctransformer_block_8/multi_head_attention_8/dense_96/Tensordot/ShapeShape&token_and_position_embedding_9/add:z:0*
T0*
_output_shapes
:2E
Ctransformer_block_8/multi_head_attention_8/dense_96/Tensordot/Shape?
Ktransformer_block_8/multi_head_attention_8/dense_96/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Ktransformer_block_8/multi_head_attention_8/dense_96/Tensordot/GatherV2/axis?
Ftransformer_block_8/multi_head_attention_8/dense_96/Tensordot/GatherV2GatherV2Ltransformer_block_8/multi_head_attention_8/dense_96/Tensordot/Shape:output:0Ktransformer_block_8/multi_head_attention_8/dense_96/Tensordot/free:output:0Ttransformer_block_8/multi_head_attention_8/dense_96/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2H
Ftransformer_block_8/multi_head_attention_8/dense_96/Tensordot/GatherV2?
Mtransformer_block_8/multi_head_attention_8/dense_96/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2O
Mtransformer_block_8/multi_head_attention_8/dense_96/Tensordot/GatherV2_1/axis?
Htransformer_block_8/multi_head_attention_8/dense_96/Tensordot/GatherV2_1GatherV2Ltransformer_block_8/multi_head_attention_8/dense_96/Tensordot/Shape:output:0Ktransformer_block_8/multi_head_attention_8/dense_96/Tensordot/axes:output:0Vtransformer_block_8/multi_head_attention_8/dense_96/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2J
Htransformer_block_8/multi_head_attention_8/dense_96/Tensordot/GatherV2_1?
Ctransformer_block_8/multi_head_attention_8/dense_96/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2E
Ctransformer_block_8/multi_head_attention_8/dense_96/Tensordot/Const?
Btransformer_block_8/multi_head_attention_8/dense_96/Tensordot/ProdProdOtransformer_block_8/multi_head_attention_8/dense_96/Tensordot/GatherV2:output:0Ltransformer_block_8/multi_head_attention_8/dense_96/Tensordot/Const:output:0*
T0*
_output_shapes
: 2D
Btransformer_block_8/multi_head_attention_8/dense_96/Tensordot/Prod?
Etransformer_block_8/multi_head_attention_8/dense_96/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2G
Etransformer_block_8/multi_head_attention_8/dense_96/Tensordot/Const_1?
Dtransformer_block_8/multi_head_attention_8/dense_96/Tensordot/Prod_1ProdQtransformer_block_8/multi_head_attention_8/dense_96/Tensordot/GatherV2_1:output:0Ntransformer_block_8/multi_head_attention_8/dense_96/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2F
Dtransformer_block_8/multi_head_attention_8/dense_96/Tensordot/Prod_1?
Itransformer_block_8/multi_head_attention_8/dense_96/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Itransformer_block_8/multi_head_attention_8/dense_96/Tensordot/concat/axis?
Dtransformer_block_8/multi_head_attention_8/dense_96/Tensordot/concatConcatV2Ktransformer_block_8/multi_head_attention_8/dense_96/Tensordot/free:output:0Ktransformer_block_8/multi_head_attention_8/dense_96/Tensordot/axes:output:0Rtransformer_block_8/multi_head_attention_8/dense_96/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2F
Dtransformer_block_8/multi_head_attention_8/dense_96/Tensordot/concat?
Ctransformer_block_8/multi_head_attention_8/dense_96/Tensordot/stackPackKtransformer_block_8/multi_head_attention_8/dense_96/Tensordot/Prod:output:0Mtransformer_block_8/multi_head_attention_8/dense_96/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2E
Ctransformer_block_8/multi_head_attention_8/dense_96/Tensordot/stack?
Gtransformer_block_8/multi_head_attention_8/dense_96/Tensordot/transpose	Transpose&token_and_position_embedding_9/add:z:0Mtransformer_block_8/multi_head_attention_8/dense_96/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????d@2I
Gtransformer_block_8/multi_head_attention_8/dense_96/Tensordot/transpose?
Etransformer_block_8/multi_head_attention_8/dense_96/Tensordot/ReshapeReshapeKtransformer_block_8/multi_head_attention_8/dense_96/Tensordot/transpose:y:0Ltransformer_block_8/multi_head_attention_8/dense_96/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2G
Etransformer_block_8/multi_head_attention_8/dense_96/Tensordot/Reshape?
Dtransformer_block_8/multi_head_attention_8/dense_96/Tensordot/MatMulMatMulNtransformer_block_8/multi_head_attention_8/dense_96/Tensordot/Reshape:output:0Ttransformer_block_8/multi_head_attention_8/dense_96/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2F
Dtransformer_block_8/multi_head_attention_8/dense_96/Tensordot/MatMul?
Etransformer_block_8/multi_head_attention_8/dense_96/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2G
Etransformer_block_8/multi_head_attention_8/dense_96/Tensordot/Const_2?
Ktransformer_block_8/multi_head_attention_8/dense_96/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Ktransformer_block_8/multi_head_attention_8/dense_96/Tensordot/concat_1/axis?
Ftransformer_block_8/multi_head_attention_8/dense_96/Tensordot/concat_1ConcatV2Otransformer_block_8/multi_head_attention_8/dense_96/Tensordot/GatherV2:output:0Ntransformer_block_8/multi_head_attention_8/dense_96/Tensordot/Const_2:output:0Ttransformer_block_8/multi_head_attention_8/dense_96/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2H
Ftransformer_block_8/multi_head_attention_8/dense_96/Tensordot/concat_1?
=transformer_block_8/multi_head_attention_8/dense_96/TensordotReshapeNtransformer_block_8/multi_head_attention_8/dense_96/Tensordot/MatMul:product:0Otransformer_block_8/multi_head_attention_8/dense_96/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????d@2?
=transformer_block_8/multi_head_attention_8/dense_96/Tensordot?
Jtransformer_block_8/multi_head_attention_8/dense_96/BiasAdd/ReadVariableOpReadVariableOpStransformer_block_8_multi_head_attention_8_dense_96_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02L
Jtransformer_block_8/multi_head_attention_8/dense_96/BiasAdd/ReadVariableOp?
;transformer_block_8/multi_head_attention_8/dense_96/BiasAddBiasAddFtransformer_block_8/multi_head_attention_8/dense_96/Tensordot:output:0Rtransformer_block_8/multi_head_attention_8/dense_96/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@2=
;transformer_block_8/multi_head_attention_8/dense_96/BiasAdd?
Ltransformer_block_8/multi_head_attention_8/dense_97/Tensordot/ReadVariableOpReadVariableOpUtransformer_block_8_multi_head_attention_8_dense_97_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02N
Ltransformer_block_8/multi_head_attention_8/dense_97/Tensordot/ReadVariableOp?
Btransformer_block_8/multi_head_attention_8/dense_97/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2D
Btransformer_block_8/multi_head_attention_8/dense_97/Tensordot/axes?
Btransformer_block_8/multi_head_attention_8/dense_97/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2D
Btransformer_block_8/multi_head_attention_8/dense_97/Tensordot/free?
Ctransformer_block_8/multi_head_attention_8/dense_97/Tensordot/ShapeShape&token_and_position_embedding_9/add:z:0*
T0*
_output_shapes
:2E
Ctransformer_block_8/multi_head_attention_8/dense_97/Tensordot/Shape?
Ktransformer_block_8/multi_head_attention_8/dense_97/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Ktransformer_block_8/multi_head_attention_8/dense_97/Tensordot/GatherV2/axis?
Ftransformer_block_8/multi_head_attention_8/dense_97/Tensordot/GatherV2GatherV2Ltransformer_block_8/multi_head_attention_8/dense_97/Tensordot/Shape:output:0Ktransformer_block_8/multi_head_attention_8/dense_97/Tensordot/free:output:0Ttransformer_block_8/multi_head_attention_8/dense_97/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2H
Ftransformer_block_8/multi_head_attention_8/dense_97/Tensordot/GatherV2?
Mtransformer_block_8/multi_head_attention_8/dense_97/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2O
Mtransformer_block_8/multi_head_attention_8/dense_97/Tensordot/GatherV2_1/axis?
Htransformer_block_8/multi_head_attention_8/dense_97/Tensordot/GatherV2_1GatherV2Ltransformer_block_8/multi_head_attention_8/dense_97/Tensordot/Shape:output:0Ktransformer_block_8/multi_head_attention_8/dense_97/Tensordot/axes:output:0Vtransformer_block_8/multi_head_attention_8/dense_97/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2J
Htransformer_block_8/multi_head_attention_8/dense_97/Tensordot/GatherV2_1?
Ctransformer_block_8/multi_head_attention_8/dense_97/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2E
Ctransformer_block_8/multi_head_attention_8/dense_97/Tensordot/Const?
Btransformer_block_8/multi_head_attention_8/dense_97/Tensordot/ProdProdOtransformer_block_8/multi_head_attention_8/dense_97/Tensordot/GatherV2:output:0Ltransformer_block_8/multi_head_attention_8/dense_97/Tensordot/Const:output:0*
T0*
_output_shapes
: 2D
Btransformer_block_8/multi_head_attention_8/dense_97/Tensordot/Prod?
Etransformer_block_8/multi_head_attention_8/dense_97/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2G
Etransformer_block_8/multi_head_attention_8/dense_97/Tensordot/Const_1?
Dtransformer_block_8/multi_head_attention_8/dense_97/Tensordot/Prod_1ProdQtransformer_block_8/multi_head_attention_8/dense_97/Tensordot/GatherV2_1:output:0Ntransformer_block_8/multi_head_attention_8/dense_97/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2F
Dtransformer_block_8/multi_head_attention_8/dense_97/Tensordot/Prod_1?
Itransformer_block_8/multi_head_attention_8/dense_97/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Itransformer_block_8/multi_head_attention_8/dense_97/Tensordot/concat/axis?
Dtransformer_block_8/multi_head_attention_8/dense_97/Tensordot/concatConcatV2Ktransformer_block_8/multi_head_attention_8/dense_97/Tensordot/free:output:0Ktransformer_block_8/multi_head_attention_8/dense_97/Tensordot/axes:output:0Rtransformer_block_8/multi_head_attention_8/dense_97/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2F
Dtransformer_block_8/multi_head_attention_8/dense_97/Tensordot/concat?
Ctransformer_block_8/multi_head_attention_8/dense_97/Tensordot/stackPackKtransformer_block_8/multi_head_attention_8/dense_97/Tensordot/Prod:output:0Mtransformer_block_8/multi_head_attention_8/dense_97/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2E
Ctransformer_block_8/multi_head_attention_8/dense_97/Tensordot/stack?
Gtransformer_block_8/multi_head_attention_8/dense_97/Tensordot/transpose	Transpose&token_and_position_embedding_9/add:z:0Mtransformer_block_8/multi_head_attention_8/dense_97/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????d@2I
Gtransformer_block_8/multi_head_attention_8/dense_97/Tensordot/transpose?
Etransformer_block_8/multi_head_attention_8/dense_97/Tensordot/ReshapeReshapeKtransformer_block_8/multi_head_attention_8/dense_97/Tensordot/transpose:y:0Ltransformer_block_8/multi_head_attention_8/dense_97/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2G
Etransformer_block_8/multi_head_attention_8/dense_97/Tensordot/Reshape?
Dtransformer_block_8/multi_head_attention_8/dense_97/Tensordot/MatMulMatMulNtransformer_block_8/multi_head_attention_8/dense_97/Tensordot/Reshape:output:0Ttransformer_block_8/multi_head_attention_8/dense_97/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2F
Dtransformer_block_8/multi_head_attention_8/dense_97/Tensordot/MatMul?
Etransformer_block_8/multi_head_attention_8/dense_97/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2G
Etransformer_block_8/multi_head_attention_8/dense_97/Tensordot/Const_2?
Ktransformer_block_8/multi_head_attention_8/dense_97/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Ktransformer_block_8/multi_head_attention_8/dense_97/Tensordot/concat_1/axis?
Ftransformer_block_8/multi_head_attention_8/dense_97/Tensordot/concat_1ConcatV2Otransformer_block_8/multi_head_attention_8/dense_97/Tensordot/GatherV2:output:0Ntransformer_block_8/multi_head_attention_8/dense_97/Tensordot/Const_2:output:0Ttransformer_block_8/multi_head_attention_8/dense_97/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2H
Ftransformer_block_8/multi_head_attention_8/dense_97/Tensordot/concat_1?
=transformer_block_8/multi_head_attention_8/dense_97/TensordotReshapeNtransformer_block_8/multi_head_attention_8/dense_97/Tensordot/MatMul:product:0Otransformer_block_8/multi_head_attention_8/dense_97/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????d@2?
=transformer_block_8/multi_head_attention_8/dense_97/Tensordot?
Jtransformer_block_8/multi_head_attention_8/dense_97/BiasAdd/ReadVariableOpReadVariableOpStransformer_block_8_multi_head_attention_8_dense_97_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02L
Jtransformer_block_8/multi_head_attention_8/dense_97/BiasAdd/ReadVariableOp?
;transformer_block_8/multi_head_attention_8/dense_97/BiasAddBiasAddFtransformer_block_8/multi_head_attention_8/dense_97/Tensordot:output:0Rtransformer_block_8/multi_head_attention_8/dense_97/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@2=
;transformer_block_8/multi_head_attention_8/dense_97/BiasAdd?
Ltransformer_block_8/multi_head_attention_8/dense_98/Tensordot/ReadVariableOpReadVariableOpUtransformer_block_8_multi_head_attention_8_dense_98_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02N
Ltransformer_block_8/multi_head_attention_8/dense_98/Tensordot/ReadVariableOp?
Btransformer_block_8/multi_head_attention_8/dense_98/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2D
Btransformer_block_8/multi_head_attention_8/dense_98/Tensordot/axes?
Btransformer_block_8/multi_head_attention_8/dense_98/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2D
Btransformer_block_8/multi_head_attention_8/dense_98/Tensordot/free?
Ctransformer_block_8/multi_head_attention_8/dense_98/Tensordot/ShapeShape&token_and_position_embedding_9/add:z:0*
T0*
_output_shapes
:2E
Ctransformer_block_8/multi_head_attention_8/dense_98/Tensordot/Shape?
Ktransformer_block_8/multi_head_attention_8/dense_98/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Ktransformer_block_8/multi_head_attention_8/dense_98/Tensordot/GatherV2/axis?
Ftransformer_block_8/multi_head_attention_8/dense_98/Tensordot/GatherV2GatherV2Ltransformer_block_8/multi_head_attention_8/dense_98/Tensordot/Shape:output:0Ktransformer_block_8/multi_head_attention_8/dense_98/Tensordot/free:output:0Ttransformer_block_8/multi_head_attention_8/dense_98/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2H
Ftransformer_block_8/multi_head_attention_8/dense_98/Tensordot/GatherV2?
Mtransformer_block_8/multi_head_attention_8/dense_98/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2O
Mtransformer_block_8/multi_head_attention_8/dense_98/Tensordot/GatherV2_1/axis?
Htransformer_block_8/multi_head_attention_8/dense_98/Tensordot/GatherV2_1GatherV2Ltransformer_block_8/multi_head_attention_8/dense_98/Tensordot/Shape:output:0Ktransformer_block_8/multi_head_attention_8/dense_98/Tensordot/axes:output:0Vtransformer_block_8/multi_head_attention_8/dense_98/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2J
Htransformer_block_8/multi_head_attention_8/dense_98/Tensordot/GatherV2_1?
Ctransformer_block_8/multi_head_attention_8/dense_98/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2E
Ctransformer_block_8/multi_head_attention_8/dense_98/Tensordot/Const?
Btransformer_block_8/multi_head_attention_8/dense_98/Tensordot/ProdProdOtransformer_block_8/multi_head_attention_8/dense_98/Tensordot/GatherV2:output:0Ltransformer_block_8/multi_head_attention_8/dense_98/Tensordot/Const:output:0*
T0*
_output_shapes
: 2D
Btransformer_block_8/multi_head_attention_8/dense_98/Tensordot/Prod?
Etransformer_block_8/multi_head_attention_8/dense_98/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2G
Etransformer_block_8/multi_head_attention_8/dense_98/Tensordot/Const_1?
Dtransformer_block_8/multi_head_attention_8/dense_98/Tensordot/Prod_1ProdQtransformer_block_8/multi_head_attention_8/dense_98/Tensordot/GatherV2_1:output:0Ntransformer_block_8/multi_head_attention_8/dense_98/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2F
Dtransformer_block_8/multi_head_attention_8/dense_98/Tensordot/Prod_1?
Itransformer_block_8/multi_head_attention_8/dense_98/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Itransformer_block_8/multi_head_attention_8/dense_98/Tensordot/concat/axis?
Dtransformer_block_8/multi_head_attention_8/dense_98/Tensordot/concatConcatV2Ktransformer_block_8/multi_head_attention_8/dense_98/Tensordot/free:output:0Ktransformer_block_8/multi_head_attention_8/dense_98/Tensordot/axes:output:0Rtransformer_block_8/multi_head_attention_8/dense_98/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2F
Dtransformer_block_8/multi_head_attention_8/dense_98/Tensordot/concat?
Ctransformer_block_8/multi_head_attention_8/dense_98/Tensordot/stackPackKtransformer_block_8/multi_head_attention_8/dense_98/Tensordot/Prod:output:0Mtransformer_block_8/multi_head_attention_8/dense_98/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2E
Ctransformer_block_8/multi_head_attention_8/dense_98/Tensordot/stack?
Gtransformer_block_8/multi_head_attention_8/dense_98/Tensordot/transpose	Transpose&token_and_position_embedding_9/add:z:0Mtransformer_block_8/multi_head_attention_8/dense_98/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????d@2I
Gtransformer_block_8/multi_head_attention_8/dense_98/Tensordot/transpose?
Etransformer_block_8/multi_head_attention_8/dense_98/Tensordot/ReshapeReshapeKtransformer_block_8/multi_head_attention_8/dense_98/Tensordot/transpose:y:0Ltransformer_block_8/multi_head_attention_8/dense_98/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2G
Etransformer_block_8/multi_head_attention_8/dense_98/Tensordot/Reshape?
Dtransformer_block_8/multi_head_attention_8/dense_98/Tensordot/MatMulMatMulNtransformer_block_8/multi_head_attention_8/dense_98/Tensordot/Reshape:output:0Ttransformer_block_8/multi_head_attention_8/dense_98/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2F
Dtransformer_block_8/multi_head_attention_8/dense_98/Tensordot/MatMul?
Etransformer_block_8/multi_head_attention_8/dense_98/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2G
Etransformer_block_8/multi_head_attention_8/dense_98/Tensordot/Const_2?
Ktransformer_block_8/multi_head_attention_8/dense_98/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Ktransformer_block_8/multi_head_attention_8/dense_98/Tensordot/concat_1/axis?
Ftransformer_block_8/multi_head_attention_8/dense_98/Tensordot/concat_1ConcatV2Otransformer_block_8/multi_head_attention_8/dense_98/Tensordot/GatherV2:output:0Ntransformer_block_8/multi_head_attention_8/dense_98/Tensordot/Const_2:output:0Ttransformer_block_8/multi_head_attention_8/dense_98/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2H
Ftransformer_block_8/multi_head_attention_8/dense_98/Tensordot/concat_1?
=transformer_block_8/multi_head_attention_8/dense_98/TensordotReshapeNtransformer_block_8/multi_head_attention_8/dense_98/Tensordot/MatMul:product:0Otransformer_block_8/multi_head_attention_8/dense_98/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????d@2?
=transformer_block_8/multi_head_attention_8/dense_98/Tensordot?
Jtransformer_block_8/multi_head_attention_8/dense_98/BiasAdd/ReadVariableOpReadVariableOpStransformer_block_8_multi_head_attention_8_dense_98_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02L
Jtransformer_block_8/multi_head_attention_8/dense_98/BiasAdd/ReadVariableOp?
;transformer_block_8/multi_head_attention_8/dense_98/BiasAddBiasAddFtransformer_block_8/multi_head_attention_8/dense_98/Tensordot:output:0Rtransformer_block_8/multi_head_attention_8/dense_98/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@2=
;transformer_block_8/multi_head_attention_8/dense_98/BiasAdd?
:transformer_block_8/multi_head_attention_8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2<
:transformer_block_8/multi_head_attention_8/Reshape/shape/1?
:transformer_block_8/multi_head_attention_8/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2<
:transformer_block_8/multi_head_attention_8/Reshape/shape/2?
:transformer_block_8/multi_head_attention_8/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2<
:transformer_block_8/multi_head_attention_8/Reshape/shape/3?
8transformer_block_8/multi_head_attention_8/Reshape/shapePackAtransformer_block_8/multi_head_attention_8/strided_slice:output:0Ctransformer_block_8/multi_head_attention_8/Reshape/shape/1:output:0Ctransformer_block_8/multi_head_attention_8/Reshape/shape/2:output:0Ctransformer_block_8/multi_head_attention_8/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2:
8transformer_block_8/multi_head_attention_8/Reshape/shape?
2transformer_block_8/multi_head_attention_8/ReshapeReshapeDtransformer_block_8/multi_head_attention_8/dense_96/BiasAdd:output:0Atransformer_block_8/multi_head_attention_8/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"?????????????????? 24
2transformer_block_8/multi_head_attention_8/Reshape?
9transformer_block_8/multi_head_attention_8/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2;
9transformer_block_8/multi_head_attention_8/transpose/perm?
4transformer_block_8/multi_head_attention_8/transpose	Transpose;transformer_block_8/multi_head_attention_8/Reshape:output:0Btransformer_block_8/multi_head_attention_8/transpose/perm:output:0*
T0*8
_output_shapes&
$:"?????????????????? 26
4transformer_block_8/multi_head_attention_8/transpose?
<transformer_block_8/multi_head_attention_8/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2>
<transformer_block_8/multi_head_attention_8/Reshape_1/shape/1?
<transformer_block_8/multi_head_attention_8/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2>
<transformer_block_8/multi_head_attention_8/Reshape_1/shape/2?
<transformer_block_8/multi_head_attention_8/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2>
<transformer_block_8/multi_head_attention_8/Reshape_1/shape/3?
:transformer_block_8/multi_head_attention_8/Reshape_1/shapePackAtransformer_block_8/multi_head_attention_8/strided_slice:output:0Etransformer_block_8/multi_head_attention_8/Reshape_1/shape/1:output:0Etransformer_block_8/multi_head_attention_8/Reshape_1/shape/2:output:0Etransformer_block_8/multi_head_attention_8/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_8/multi_head_attention_8/Reshape_1/shape?
4transformer_block_8/multi_head_attention_8/Reshape_1ReshapeDtransformer_block_8/multi_head_attention_8/dense_97/BiasAdd:output:0Ctransformer_block_8/multi_head_attention_8/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"?????????????????? 26
4transformer_block_8/multi_head_attention_8/Reshape_1?
;transformer_block_8/multi_head_attention_8/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2=
;transformer_block_8/multi_head_attention_8/transpose_1/perm?
6transformer_block_8/multi_head_attention_8/transpose_1	Transpose=transformer_block_8/multi_head_attention_8/Reshape_1:output:0Dtransformer_block_8/multi_head_attention_8/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"?????????????????? 28
6transformer_block_8/multi_head_attention_8/transpose_1?
<transformer_block_8/multi_head_attention_8/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2>
<transformer_block_8/multi_head_attention_8/Reshape_2/shape/1?
<transformer_block_8/multi_head_attention_8/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2>
<transformer_block_8/multi_head_attention_8/Reshape_2/shape/2?
<transformer_block_8/multi_head_attention_8/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2>
<transformer_block_8/multi_head_attention_8/Reshape_2/shape/3?
:transformer_block_8/multi_head_attention_8/Reshape_2/shapePackAtransformer_block_8/multi_head_attention_8/strided_slice:output:0Etransformer_block_8/multi_head_attention_8/Reshape_2/shape/1:output:0Etransformer_block_8/multi_head_attention_8/Reshape_2/shape/2:output:0Etransformer_block_8/multi_head_attention_8/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_8/multi_head_attention_8/Reshape_2/shape?
4transformer_block_8/multi_head_attention_8/Reshape_2ReshapeDtransformer_block_8/multi_head_attention_8/dense_98/BiasAdd:output:0Ctransformer_block_8/multi_head_attention_8/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"?????????????????? 26
4transformer_block_8/multi_head_attention_8/Reshape_2?
;transformer_block_8/multi_head_attention_8/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2=
;transformer_block_8/multi_head_attention_8/transpose_2/perm?
6transformer_block_8/multi_head_attention_8/transpose_2	Transpose=transformer_block_8/multi_head_attention_8/Reshape_2:output:0Dtransformer_block_8/multi_head_attention_8/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"?????????????????? 28
6transformer_block_8/multi_head_attention_8/transpose_2?
1transformer_block_8/multi_head_attention_8/MatMulBatchMatMulV28transformer_block_8/multi_head_attention_8/transpose:y:0:transformer_block_8/multi_head_attention_8/transpose_1:y:0*
T0*A
_output_shapes/
-:+???????????????????????????*
adj_y(23
1transformer_block_8/multi_head_attention_8/MatMul?
2transformer_block_8/multi_head_attention_8/Shape_1Shape:transformer_block_8/multi_head_attention_8/transpose_1:y:0*
T0*
_output_shapes
:24
2transformer_block_8/multi_head_attention_8/Shape_1?
@transformer_block_8/multi_head_attention_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2B
@transformer_block_8/multi_head_attention_8/strided_slice_1/stack?
Btransformer_block_8/multi_head_attention_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2D
Btransformer_block_8/multi_head_attention_8/strided_slice_1/stack_1?
Btransformer_block_8/multi_head_attention_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
Btransformer_block_8/multi_head_attention_8/strided_slice_1/stack_2?
:transformer_block_8/multi_head_attention_8/strided_slice_1StridedSlice;transformer_block_8/multi_head_attention_8/Shape_1:output:0Itransformer_block_8/multi_head_attention_8/strided_slice_1/stack:output:0Ktransformer_block_8/multi_head_attention_8/strided_slice_1/stack_1:output:0Ktransformer_block_8/multi_head_attention_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2<
:transformer_block_8/multi_head_attention_8/strided_slice_1?
/transformer_block_8/multi_head_attention_8/CastCastCtransformer_block_8/multi_head_attention_8/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 21
/transformer_block_8/multi_head_attention_8/Cast?
/transformer_block_8/multi_head_attention_8/SqrtSqrt3transformer_block_8/multi_head_attention_8/Cast:y:0*
T0*
_output_shapes
: 21
/transformer_block_8/multi_head_attention_8/Sqrt?
2transformer_block_8/multi_head_attention_8/truedivRealDiv:transformer_block_8/multi_head_attention_8/MatMul:output:03transformer_block_8/multi_head_attention_8/Sqrt:y:0*
T0*A
_output_shapes/
-:+???????????????????????????24
2transformer_block_8/multi_head_attention_8/truediv?
2transformer_block_8/multi_head_attention_8/SoftmaxSoftmax6transformer_block_8/multi_head_attention_8/truediv:z:0*
T0*A
_output_shapes/
-:+???????????????????????????24
2transformer_block_8/multi_head_attention_8/Softmax?
3transformer_block_8/multi_head_attention_8/MatMul_1BatchMatMulV2<transformer_block_8/multi_head_attention_8/Softmax:softmax:0:transformer_block_8/multi_head_attention_8/transpose_2:y:0*
T0*8
_output_shapes&
$:"?????????????????? 25
3transformer_block_8/multi_head_attention_8/MatMul_1?
;transformer_block_8/multi_head_attention_8/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             2=
;transformer_block_8/multi_head_attention_8/transpose_3/perm?
6transformer_block_8/multi_head_attention_8/transpose_3	Transpose<transformer_block_8/multi_head_attention_8/MatMul_1:output:0Dtransformer_block_8/multi_head_attention_8/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"?????????????????? 28
6transformer_block_8/multi_head_attention_8/transpose_3?
<transformer_block_8/multi_head_attention_8/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2>
<transformer_block_8/multi_head_attention_8/Reshape_3/shape/1?
<transformer_block_8/multi_head_attention_8/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@2>
<transformer_block_8/multi_head_attention_8/Reshape_3/shape/2?
:transformer_block_8/multi_head_attention_8/Reshape_3/shapePackAtransformer_block_8/multi_head_attention_8/strided_slice:output:0Etransformer_block_8/multi_head_attention_8/Reshape_3/shape/1:output:0Etransformer_block_8/multi_head_attention_8/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_8/multi_head_attention_8/Reshape_3/shape?
4transformer_block_8/multi_head_attention_8/Reshape_3Reshape:transformer_block_8/multi_head_attention_8/transpose_3:y:0Ctransformer_block_8/multi_head_attention_8/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :??????????????????@26
4transformer_block_8/multi_head_attention_8/Reshape_3?
Ltransformer_block_8/multi_head_attention_8/dense_99/Tensordot/ReadVariableOpReadVariableOpUtransformer_block_8_multi_head_attention_8_dense_99_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02N
Ltransformer_block_8/multi_head_attention_8/dense_99/Tensordot/ReadVariableOp?
Btransformer_block_8/multi_head_attention_8/dense_99/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2D
Btransformer_block_8/multi_head_attention_8/dense_99/Tensordot/axes?
Btransformer_block_8/multi_head_attention_8/dense_99/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2D
Btransformer_block_8/multi_head_attention_8/dense_99/Tensordot/free?
Ctransformer_block_8/multi_head_attention_8/dense_99/Tensordot/ShapeShape=transformer_block_8/multi_head_attention_8/Reshape_3:output:0*
T0*
_output_shapes
:2E
Ctransformer_block_8/multi_head_attention_8/dense_99/Tensordot/Shape?
Ktransformer_block_8/multi_head_attention_8/dense_99/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Ktransformer_block_8/multi_head_attention_8/dense_99/Tensordot/GatherV2/axis?
Ftransformer_block_8/multi_head_attention_8/dense_99/Tensordot/GatherV2GatherV2Ltransformer_block_8/multi_head_attention_8/dense_99/Tensordot/Shape:output:0Ktransformer_block_8/multi_head_attention_8/dense_99/Tensordot/free:output:0Ttransformer_block_8/multi_head_attention_8/dense_99/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2H
Ftransformer_block_8/multi_head_attention_8/dense_99/Tensordot/GatherV2?
Mtransformer_block_8/multi_head_attention_8/dense_99/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2O
Mtransformer_block_8/multi_head_attention_8/dense_99/Tensordot/GatherV2_1/axis?
Htransformer_block_8/multi_head_attention_8/dense_99/Tensordot/GatherV2_1GatherV2Ltransformer_block_8/multi_head_attention_8/dense_99/Tensordot/Shape:output:0Ktransformer_block_8/multi_head_attention_8/dense_99/Tensordot/axes:output:0Vtransformer_block_8/multi_head_attention_8/dense_99/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2J
Htransformer_block_8/multi_head_attention_8/dense_99/Tensordot/GatherV2_1?
Ctransformer_block_8/multi_head_attention_8/dense_99/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2E
Ctransformer_block_8/multi_head_attention_8/dense_99/Tensordot/Const?
Btransformer_block_8/multi_head_attention_8/dense_99/Tensordot/ProdProdOtransformer_block_8/multi_head_attention_8/dense_99/Tensordot/GatherV2:output:0Ltransformer_block_8/multi_head_attention_8/dense_99/Tensordot/Const:output:0*
T0*
_output_shapes
: 2D
Btransformer_block_8/multi_head_attention_8/dense_99/Tensordot/Prod?
Etransformer_block_8/multi_head_attention_8/dense_99/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2G
Etransformer_block_8/multi_head_attention_8/dense_99/Tensordot/Const_1?
Dtransformer_block_8/multi_head_attention_8/dense_99/Tensordot/Prod_1ProdQtransformer_block_8/multi_head_attention_8/dense_99/Tensordot/GatherV2_1:output:0Ntransformer_block_8/multi_head_attention_8/dense_99/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2F
Dtransformer_block_8/multi_head_attention_8/dense_99/Tensordot/Prod_1?
Itransformer_block_8/multi_head_attention_8/dense_99/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Itransformer_block_8/multi_head_attention_8/dense_99/Tensordot/concat/axis?
Dtransformer_block_8/multi_head_attention_8/dense_99/Tensordot/concatConcatV2Ktransformer_block_8/multi_head_attention_8/dense_99/Tensordot/free:output:0Ktransformer_block_8/multi_head_attention_8/dense_99/Tensordot/axes:output:0Rtransformer_block_8/multi_head_attention_8/dense_99/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2F
Dtransformer_block_8/multi_head_attention_8/dense_99/Tensordot/concat?
Ctransformer_block_8/multi_head_attention_8/dense_99/Tensordot/stackPackKtransformer_block_8/multi_head_attention_8/dense_99/Tensordot/Prod:output:0Mtransformer_block_8/multi_head_attention_8/dense_99/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2E
Ctransformer_block_8/multi_head_attention_8/dense_99/Tensordot/stack?
Gtransformer_block_8/multi_head_attention_8/dense_99/Tensordot/transpose	Transpose=transformer_block_8/multi_head_attention_8/Reshape_3:output:0Mtransformer_block_8/multi_head_attention_8/dense_99/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :??????????????????@2I
Gtransformer_block_8/multi_head_attention_8/dense_99/Tensordot/transpose?
Etransformer_block_8/multi_head_attention_8/dense_99/Tensordot/ReshapeReshapeKtransformer_block_8/multi_head_attention_8/dense_99/Tensordot/transpose:y:0Ltransformer_block_8/multi_head_attention_8/dense_99/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2G
Etransformer_block_8/multi_head_attention_8/dense_99/Tensordot/Reshape?
Dtransformer_block_8/multi_head_attention_8/dense_99/Tensordot/MatMulMatMulNtransformer_block_8/multi_head_attention_8/dense_99/Tensordot/Reshape:output:0Ttransformer_block_8/multi_head_attention_8/dense_99/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2F
Dtransformer_block_8/multi_head_attention_8/dense_99/Tensordot/MatMul?
Etransformer_block_8/multi_head_attention_8/dense_99/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2G
Etransformer_block_8/multi_head_attention_8/dense_99/Tensordot/Const_2?
Ktransformer_block_8/multi_head_attention_8/dense_99/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Ktransformer_block_8/multi_head_attention_8/dense_99/Tensordot/concat_1/axis?
Ftransformer_block_8/multi_head_attention_8/dense_99/Tensordot/concat_1ConcatV2Otransformer_block_8/multi_head_attention_8/dense_99/Tensordot/GatherV2:output:0Ntransformer_block_8/multi_head_attention_8/dense_99/Tensordot/Const_2:output:0Ttransformer_block_8/multi_head_attention_8/dense_99/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2H
Ftransformer_block_8/multi_head_attention_8/dense_99/Tensordot/concat_1?
=transformer_block_8/multi_head_attention_8/dense_99/TensordotReshapeNtransformer_block_8/multi_head_attention_8/dense_99/Tensordot/MatMul:product:0Otransformer_block_8/multi_head_attention_8/dense_99/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :??????????????????@2?
=transformer_block_8/multi_head_attention_8/dense_99/Tensordot?
Jtransformer_block_8/multi_head_attention_8/dense_99/BiasAdd/ReadVariableOpReadVariableOpStransformer_block_8_multi_head_attention_8_dense_99_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02L
Jtransformer_block_8/multi_head_attention_8/dense_99/BiasAdd/ReadVariableOp?
;transformer_block_8/multi_head_attention_8/dense_99/BiasAddBiasAddFtransformer_block_8/multi_head_attention_8/dense_99/Tensordot:output:0Rtransformer_block_8/multi_head_attention_8/dense_99/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????@2=
;transformer_block_8/multi_head_attention_8/dense_99/BiasAdd?
'transformer_block_8/dropout_42/IdentityIdentityDtransformer_block_8/multi_head_attention_8/dense_99/BiasAdd:output:0*
T0*4
_output_shapes"
 :??????????????????@2)
'transformer_block_8/dropout_42/Identity?
transformer_block_8/addAddV2&token_and_position_embedding_9/add:z:00transformer_block_8/dropout_42/Identity:output:0*
T0*+
_output_shapes
:?????????d@2
transformer_block_8/add?
Itransformer_block_8/layer_normalization_26/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2K
Itransformer_block_8/layer_normalization_26/moments/mean/reduction_indices?
7transformer_block_8/layer_normalization_26/moments/meanMeantransformer_block_8/add:z:0Rtransformer_block_8/layer_normalization_26/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????d*
	keep_dims(29
7transformer_block_8/layer_normalization_26/moments/mean?
?transformer_block_8/layer_normalization_26/moments/StopGradientStopGradient@transformer_block_8/layer_normalization_26/moments/mean:output:0*
T0*+
_output_shapes
:?????????d2A
?transformer_block_8/layer_normalization_26/moments/StopGradient?
Dtransformer_block_8/layer_normalization_26/moments/SquaredDifferenceSquaredDifferencetransformer_block_8/add:z:0Htransformer_block_8/layer_normalization_26/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????d@2F
Dtransformer_block_8/layer_normalization_26/moments/SquaredDifference?
Mtransformer_block_8/layer_normalization_26/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2O
Mtransformer_block_8/layer_normalization_26/moments/variance/reduction_indices?
;transformer_block_8/layer_normalization_26/moments/varianceMeanHtransformer_block_8/layer_normalization_26/moments/SquaredDifference:z:0Vtransformer_block_8/layer_normalization_26/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????d*
	keep_dims(2=
;transformer_block_8/layer_normalization_26/moments/variance?
:transformer_block_8/layer_normalization_26/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52<
:transformer_block_8/layer_normalization_26/batchnorm/add/y?
8transformer_block_8/layer_normalization_26/batchnorm/addAddV2Dtransformer_block_8/layer_normalization_26/moments/variance:output:0Ctransformer_block_8/layer_normalization_26/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????d2:
8transformer_block_8/layer_normalization_26/batchnorm/add?
:transformer_block_8/layer_normalization_26/batchnorm/RsqrtRsqrt<transformer_block_8/layer_normalization_26/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????d2<
:transformer_block_8/layer_normalization_26/batchnorm/Rsqrt?
Gtransformer_block_8/layer_normalization_26/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_8_layer_normalization_26_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02I
Gtransformer_block_8/layer_normalization_26/batchnorm/mul/ReadVariableOp?
8transformer_block_8/layer_normalization_26/batchnorm/mulMul>transformer_block_8/layer_normalization_26/batchnorm/Rsqrt:y:0Otransformer_block_8/layer_normalization_26/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@2:
8transformer_block_8/layer_normalization_26/batchnorm/mul?
:transformer_block_8/layer_normalization_26/batchnorm/mul_1Multransformer_block_8/add:z:0<transformer_block_8/layer_normalization_26/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????d@2<
:transformer_block_8/layer_normalization_26/batchnorm/mul_1?
:transformer_block_8/layer_normalization_26/batchnorm/mul_2Mul@transformer_block_8/layer_normalization_26/moments/mean:output:0<transformer_block_8/layer_normalization_26/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????d@2<
:transformer_block_8/layer_normalization_26/batchnorm/mul_2?
Ctransformer_block_8/layer_normalization_26/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_8_layer_normalization_26_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02E
Ctransformer_block_8/layer_normalization_26/batchnorm/ReadVariableOp?
8transformer_block_8/layer_normalization_26/batchnorm/subSubKtransformer_block_8/layer_normalization_26/batchnorm/ReadVariableOp:value:0>transformer_block_8/layer_normalization_26/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????d@2:
8transformer_block_8/layer_normalization_26/batchnorm/sub?
:transformer_block_8/layer_normalization_26/batchnorm/add_1AddV2>transformer_block_8/layer_normalization_26/batchnorm/mul_1:z:0<transformer_block_8/layer_normalization_26/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????d@2<
:transformer_block_8/layer_normalization_26/batchnorm/add_1?
Ctransformer_block_8/sequential_8/dense_100/Tensordot/ReadVariableOpReadVariableOpLtransformer_block_8_sequential_8_dense_100_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02E
Ctransformer_block_8/sequential_8/dense_100/Tensordot/ReadVariableOp?
9transformer_block_8/sequential_8/dense_100/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2;
9transformer_block_8/sequential_8/dense_100/Tensordot/axes?
9transformer_block_8/sequential_8/dense_100/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2;
9transformer_block_8/sequential_8/dense_100/Tensordot/free?
:transformer_block_8/sequential_8/dense_100/Tensordot/ShapeShape>transformer_block_8/layer_normalization_26/batchnorm/add_1:z:0*
T0*
_output_shapes
:2<
:transformer_block_8/sequential_8/dense_100/Tensordot/Shape?
Btransformer_block_8/sequential_8/dense_100/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block_8/sequential_8/dense_100/Tensordot/GatherV2/axis?
=transformer_block_8/sequential_8/dense_100/Tensordot/GatherV2GatherV2Ctransformer_block_8/sequential_8/dense_100/Tensordot/Shape:output:0Btransformer_block_8/sequential_8/dense_100/Tensordot/free:output:0Ktransformer_block_8/sequential_8/dense_100/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2?
=transformer_block_8/sequential_8/dense_100/Tensordot/GatherV2?
Dtransformer_block_8/sequential_8/dense_100/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2F
Dtransformer_block_8/sequential_8/dense_100/Tensordot/GatherV2_1/axis?
?transformer_block_8/sequential_8/dense_100/Tensordot/GatherV2_1GatherV2Ctransformer_block_8/sequential_8/dense_100/Tensordot/Shape:output:0Btransformer_block_8/sequential_8/dense_100/Tensordot/axes:output:0Mtransformer_block_8/sequential_8/dense_100/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2A
?transformer_block_8/sequential_8/dense_100/Tensordot/GatherV2_1?
:transformer_block_8/sequential_8/dense_100/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block_8/sequential_8/dense_100/Tensordot/Const?
9transformer_block_8/sequential_8/dense_100/Tensordot/ProdProdFtransformer_block_8/sequential_8/dense_100/Tensordot/GatherV2:output:0Ctransformer_block_8/sequential_8/dense_100/Tensordot/Const:output:0*
T0*
_output_shapes
: 2;
9transformer_block_8/sequential_8/dense_100/Tensordot/Prod?
<transformer_block_8/sequential_8/dense_100/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<transformer_block_8/sequential_8/dense_100/Tensordot/Const_1?
;transformer_block_8/sequential_8/dense_100/Tensordot/Prod_1ProdHtransformer_block_8/sequential_8/dense_100/Tensordot/GatherV2_1:output:0Etransformer_block_8/sequential_8/dense_100/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2=
;transformer_block_8/sequential_8/dense_100/Tensordot/Prod_1?
@transformer_block_8/sequential_8/dense_100/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_8/sequential_8/dense_100/Tensordot/concat/axis?
;transformer_block_8/sequential_8/dense_100/Tensordot/concatConcatV2Btransformer_block_8/sequential_8/dense_100/Tensordot/free:output:0Btransformer_block_8/sequential_8/dense_100/Tensordot/axes:output:0Itransformer_block_8/sequential_8/dense_100/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block_8/sequential_8/dense_100/Tensordot/concat?
:transformer_block_8/sequential_8/dense_100/Tensordot/stackPackBtransformer_block_8/sequential_8/dense_100/Tensordot/Prod:output:0Dtransformer_block_8/sequential_8/dense_100/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_8/sequential_8/dense_100/Tensordot/stack?
>transformer_block_8/sequential_8/dense_100/Tensordot/transpose	Transpose>transformer_block_8/layer_normalization_26/batchnorm/add_1:z:0Dtransformer_block_8/sequential_8/dense_100/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????d@2@
>transformer_block_8/sequential_8/dense_100/Tensordot/transpose?
<transformer_block_8/sequential_8/dense_100/Tensordot/ReshapeReshapeBtransformer_block_8/sequential_8/dense_100/Tensordot/transpose:y:0Ctransformer_block_8/sequential_8/dense_100/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2>
<transformer_block_8/sequential_8/dense_100/Tensordot/Reshape?
;transformer_block_8/sequential_8/dense_100/Tensordot/MatMulMatMulEtransformer_block_8/sequential_8/dense_100/Tensordot/Reshape:output:0Ktransformer_block_8/sequential_8/dense_100/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2=
;transformer_block_8/sequential_8/dense_100/Tensordot/MatMul?
<transformer_block_8/sequential_8/dense_100/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2>
<transformer_block_8/sequential_8/dense_100/Tensordot/Const_2?
Btransformer_block_8/sequential_8/dense_100/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block_8/sequential_8/dense_100/Tensordot/concat_1/axis?
=transformer_block_8/sequential_8/dense_100/Tensordot/concat_1ConcatV2Ftransformer_block_8/sequential_8/dense_100/Tensordot/GatherV2:output:0Etransformer_block_8/sequential_8/dense_100/Tensordot/Const_2:output:0Ktransformer_block_8/sequential_8/dense_100/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2?
=transformer_block_8/sequential_8/dense_100/Tensordot/concat_1?
4transformer_block_8/sequential_8/dense_100/TensordotReshapeEtransformer_block_8/sequential_8/dense_100/Tensordot/MatMul:product:0Ftransformer_block_8/sequential_8/dense_100/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????d 26
4transformer_block_8/sequential_8/dense_100/Tensordot?
Atransformer_block_8/sequential_8/dense_100/BiasAdd/ReadVariableOpReadVariableOpJtransformer_block_8_sequential_8_dense_100_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02C
Atransformer_block_8/sequential_8/dense_100/BiasAdd/ReadVariableOp?
2transformer_block_8/sequential_8/dense_100/BiasAddBiasAdd=transformer_block_8/sequential_8/dense_100/Tensordot:output:0Itransformer_block_8/sequential_8/dense_100/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d 24
2transformer_block_8/sequential_8/dense_100/BiasAdd?
/transformer_block_8/sequential_8/dense_100/ReluRelu;transformer_block_8/sequential_8/dense_100/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d 21
/transformer_block_8/sequential_8/dense_100/Relu?
Ctransformer_block_8/sequential_8/dense_101/Tensordot/ReadVariableOpReadVariableOpLtransformer_block_8_sequential_8_dense_101_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02E
Ctransformer_block_8/sequential_8/dense_101/Tensordot/ReadVariableOp?
9transformer_block_8/sequential_8/dense_101/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2;
9transformer_block_8/sequential_8/dense_101/Tensordot/axes?
9transformer_block_8/sequential_8/dense_101/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2;
9transformer_block_8/sequential_8/dense_101/Tensordot/free?
:transformer_block_8/sequential_8/dense_101/Tensordot/ShapeShape=transformer_block_8/sequential_8/dense_100/Relu:activations:0*
T0*
_output_shapes
:2<
:transformer_block_8/sequential_8/dense_101/Tensordot/Shape?
Btransformer_block_8/sequential_8/dense_101/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block_8/sequential_8/dense_101/Tensordot/GatherV2/axis?
=transformer_block_8/sequential_8/dense_101/Tensordot/GatherV2GatherV2Ctransformer_block_8/sequential_8/dense_101/Tensordot/Shape:output:0Btransformer_block_8/sequential_8/dense_101/Tensordot/free:output:0Ktransformer_block_8/sequential_8/dense_101/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2?
=transformer_block_8/sequential_8/dense_101/Tensordot/GatherV2?
Dtransformer_block_8/sequential_8/dense_101/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2F
Dtransformer_block_8/sequential_8/dense_101/Tensordot/GatherV2_1/axis?
?transformer_block_8/sequential_8/dense_101/Tensordot/GatherV2_1GatherV2Ctransformer_block_8/sequential_8/dense_101/Tensordot/Shape:output:0Btransformer_block_8/sequential_8/dense_101/Tensordot/axes:output:0Mtransformer_block_8/sequential_8/dense_101/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2A
?transformer_block_8/sequential_8/dense_101/Tensordot/GatherV2_1?
:transformer_block_8/sequential_8/dense_101/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2<
:transformer_block_8/sequential_8/dense_101/Tensordot/Const?
9transformer_block_8/sequential_8/dense_101/Tensordot/ProdProdFtransformer_block_8/sequential_8/dense_101/Tensordot/GatherV2:output:0Ctransformer_block_8/sequential_8/dense_101/Tensordot/Const:output:0*
T0*
_output_shapes
: 2;
9transformer_block_8/sequential_8/dense_101/Tensordot/Prod?
<transformer_block_8/sequential_8/dense_101/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<transformer_block_8/sequential_8/dense_101/Tensordot/Const_1?
;transformer_block_8/sequential_8/dense_101/Tensordot/Prod_1ProdHtransformer_block_8/sequential_8/dense_101/Tensordot/GatherV2_1:output:0Etransformer_block_8/sequential_8/dense_101/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2=
;transformer_block_8/sequential_8/dense_101/Tensordot/Prod_1?
@transformer_block_8/sequential_8/dense_101/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2B
@transformer_block_8/sequential_8/dense_101/Tensordot/concat/axis?
;transformer_block_8/sequential_8/dense_101/Tensordot/concatConcatV2Btransformer_block_8/sequential_8/dense_101/Tensordot/free:output:0Btransformer_block_8/sequential_8/dense_101/Tensordot/axes:output:0Itransformer_block_8/sequential_8/dense_101/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block_8/sequential_8/dense_101/Tensordot/concat?
:transformer_block_8/sequential_8/dense_101/Tensordot/stackPackBtransformer_block_8/sequential_8/dense_101/Tensordot/Prod:output:0Dtransformer_block_8/sequential_8/dense_101/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2<
:transformer_block_8/sequential_8/dense_101/Tensordot/stack?
>transformer_block_8/sequential_8/dense_101/Tensordot/transpose	Transpose=transformer_block_8/sequential_8/dense_100/Relu:activations:0Dtransformer_block_8/sequential_8/dense_101/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????d 2@
>transformer_block_8/sequential_8/dense_101/Tensordot/transpose?
<transformer_block_8/sequential_8/dense_101/Tensordot/ReshapeReshapeBtransformer_block_8/sequential_8/dense_101/Tensordot/transpose:y:0Ctransformer_block_8/sequential_8/dense_101/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2>
<transformer_block_8/sequential_8/dense_101/Tensordot/Reshape?
;transformer_block_8/sequential_8/dense_101/Tensordot/MatMulMatMulEtransformer_block_8/sequential_8/dense_101/Tensordot/Reshape:output:0Ktransformer_block_8/sequential_8/dense_101/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2=
;transformer_block_8/sequential_8/dense_101/Tensordot/MatMul?
<transformer_block_8/sequential_8/dense_101/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2>
<transformer_block_8/sequential_8/dense_101/Tensordot/Const_2?
Btransformer_block_8/sequential_8/dense_101/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2D
Btransformer_block_8/sequential_8/dense_101/Tensordot/concat_1/axis?
=transformer_block_8/sequential_8/dense_101/Tensordot/concat_1ConcatV2Ftransformer_block_8/sequential_8/dense_101/Tensordot/GatherV2:output:0Etransformer_block_8/sequential_8/dense_101/Tensordot/Const_2:output:0Ktransformer_block_8/sequential_8/dense_101/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2?
=transformer_block_8/sequential_8/dense_101/Tensordot/concat_1?
4transformer_block_8/sequential_8/dense_101/TensordotReshapeEtransformer_block_8/sequential_8/dense_101/Tensordot/MatMul:product:0Ftransformer_block_8/sequential_8/dense_101/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????d@26
4transformer_block_8/sequential_8/dense_101/Tensordot?
Atransformer_block_8/sequential_8/dense_101/BiasAdd/ReadVariableOpReadVariableOpJtransformer_block_8_sequential_8_dense_101_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02C
Atransformer_block_8/sequential_8/dense_101/BiasAdd/ReadVariableOp?
2transformer_block_8/sequential_8/dense_101/BiasAddBiasAdd=transformer_block_8/sequential_8/dense_101/Tensordot:output:0Itransformer_block_8/sequential_8/dense_101/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@24
2transformer_block_8/sequential_8/dense_101/BiasAdd?
'transformer_block_8/dropout_43/IdentityIdentity;transformer_block_8/sequential_8/dense_101/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d@2)
'transformer_block_8/dropout_43/Identity?
transformer_block_8/add_1AddV2>transformer_block_8/layer_normalization_26/batchnorm/add_1:z:00transformer_block_8/dropout_43/Identity:output:0*
T0*+
_output_shapes
:?????????d@2
transformer_block_8/add_1?
Itransformer_block_8/layer_normalization_27/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2K
Itransformer_block_8/layer_normalization_27/moments/mean/reduction_indices?
7transformer_block_8/layer_normalization_27/moments/meanMeantransformer_block_8/add_1:z:0Rtransformer_block_8/layer_normalization_27/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????d*
	keep_dims(29
7transformer_block_8/layer_normalization_27/moments/mean?
?transformer_block_8/layer_normalization_27/moments/StopGradientStopGradient@transformer_block_8/layer_normalization_27/moments/mean:output:0*
T0*+
_output_shapes
:?????????d2A
?transformer_block_8/layer_normalization_27/moments/StopGradient?
Dtransformer_block_8/layer_normalization_27/moments/SquaredDifferenceSquaredDifferencetransformer_block_8/add_1:z:0Htransformer_block_8/layer_normalization_27/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????d@2F
Dtransformer_block_8/layer_normalization_27/moments/SquaredDifference?
Mtransformer_block_8/layer_normalization_27/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2O
Mtransformer_block_8/layer_normalization_27/moments/variance/reduction_indices?
;transformer_block_8/layer_normalization_27/moments/varianceMeanHtransformer_block_8/layer_normalization_27/moments/SquaredDifference:z:0Vtransformer_block_8/layer_normalization_27/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????d*
	keep_dims(2=
;transformer_block_8/layer_normalization_27/moments/variance?
:transformer_block_8/layer_normalization_27/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52<
:transformer_block_8/layer_normalization_27/batchnorm/add/y?
8transformer_block_8/layer_normalization_27/batchnorm/addAddV2Dtransformer_block_8/layer_normalization_27/moments/variance:output:0Ctransformer_block_8/layer_normalization_27/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????d2:
8transformer_block_8/layer_normalization_27/batchnorm/add?
:transformer_block_8/layer_normalization_27/batchnorm/RsqrtRsqrt<transformer_block_8/layer_normalization_27/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????d2<
:transformer_block_8/layer_normalization_27/batchnorm/Rsqrt?
Gtransformer_block_8/layer_normalization_27/batchnorm/mul/ReadVariableOpReadVariableOpPtransformer_block_8_layer_normalization_27_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02I
Gtransformer_block_8/layer_normalization_27/batchnorm/mul/ReadVariableOp?
8transformer_block_8/layer_normalization_27/batchnorm/mulMul>transformer_block_8/layer_normalization_27/batchnorm/Rsqrt:y:0Otransformer_block_8/layer_normalization_27/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@2:
8transformer_block_8/layer_normalization_27/batchnorm/mul?
:transformer_block_8/layer_normalization_27/batchnorm/mul_1Multransformer_block_8/add_1:z:0<transformer_block_8/layer_normalization_27/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????d@2<
:transformer_block_8/layer_normalization_27/batchnorm/mul_1?
:transformer_block_8/layer_normalization_27/batchnorm/mul_2Mul@transformer_block_8/layer_normalization_27/moments/mean:output:0<transformer_block_8/layer_normalization_27/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????d@2<
:transformer_block_8/layer_normalization_27/batchnorm/mul_2?
Ctransformer_block_8/layer_normalization_27/batchnorm/ReadVariableOpReadVariableOpLtransformer_block_8_layer_normalization_27_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02E
Ctransformer_block_8/layer_normalization_27/batchnorm/ReadVariableOp?
8transformer_block_8/layer_normalization_27/batchnorm/subSubKtransformer_block_8/layer_normalization_27/batchnorm/ReadVariableOp:value:0>transformer_block_8/layer_normalization_27/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????d@2:
8transformer_block_8/layer_normalization_27/batchnorm/sub?
:transformer_block_8/layer_normalization_27/batchnorm/add_1AddV2>transformer_block_8/layer_normalization_27/batchnorm/mul_1:z:0<transformer_block_8/layer_normalization_27/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????d@2<
:transformer_block_8/layer_normalization_27/batchnorm/add_1?
1global_average_pooling1d_8/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :23
1global_average_pooling1d_8/Mean/reduction_indices?
global_average_pooling1d_8/MeanMean>transformer_block_8/layer_normalization_27/batchnorm/add_1:z:0:global_average_pooling1d_8/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@2!
global_average_pooling1d_8/Mean?
dropout_44/IdentityIdentity(global_average_pooling1d_8/Mean:output:0*
T0*'
_output_shapes
:?????????@2
dropout_44/Identity?
dense_102/MatMul/ReadVariableOpReadVariableOp(dense_102_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02!
dense_102/MatMul/ReadVariableOp?
dense_102/MatMulMatMuldropout_44/Identity:output:0'dense_102/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_102/MatMul?
 dense_102/BiasAdd/ReadVariableOpReadVariableOp)dense_102_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_102/BiasAdd/ReadVariableOp?
dense_102/BiasAddBiasAdddense_102/MatMul:product:0(dense_102/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_102/BiasAddv
dense_102/ReluReludense_102/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_102/Relu?
dropout_45/IdentityIdentitydense_102/Relu:activations:0*
T0*'
_output_shapes
:?????????2
dropout_45/Identity?
dense_103/MatMul/ReadVariableOpReadVariableOp(dense_103_matmul_readvariableop_resource*
_output_shapes

:*
dtype02!
dense_103/MatMul/ReadVariableOp?
dense_103/MatMulMatMuldropout_45/Identity:output:0'dense_103/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_103/MatMul?
 dense_103/BiasAdd/ReadVariableOpReadVariableOp)dense_103_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_103/BiasAdd/ReadVariableOp?
dense_103/BiasAddBiasAdddense_103/MatMul:product:0(dense_103/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_103/BiasAdd
dense_103/SoftmaxSoftmaxdense_103/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_103/Softmax?
IdentityIdentitydense_103/Softmax:softmax:0!^dense_102/BiasAdd/ReadVariableOp ^dense_102/MatMul/ReadVariableOp!^dense_103/BiasAdd/ReadVariableOp ^dense_103/MatMul/ReadVariableOp=^token_and_position_embedding_9/embedding_20/embedding_lookup=^token_and_position_embedding_9/embedding_21/embedding_lookupD^transformer_block_8/layer_normalization_26/batchnorm/ReadVariableOpH^transformer_block_8/layer_normalization_26/batchnorm/mul/ReadVariableOpD^transformer_block_8/layer_normalization_27/batchnorm/ReadVariableOpH^transformer_block_8/layer_normalization_27/batchnorm/mul/ReadVariableOpK^transformer_block_8/multi_head_attention_8/dense_96/BiasAdd/ReadVariableOpM^transformer_block_8/multi_head_attention_8/dense_96/Tensordot/ReadVariableOpK^transformer_block_8/multi_head_attention_8/dense_97/BiasAdd/ReadVariableOpM^transformer_block_8/multi_head_attention_8/dense_97/Tensordot/ReadVariableOpK^transformer_block_8/multi_head_attention_8/dense_98/BiasAdd/ReadVariableOpM^transformer_block_8/multi_head_attention_8/dense_98/Tensordot/ReadVariableOpK^transformer_block_8/multi_head_attention_8/dense_99/BiasAdd/ReadVariableOpM^transformer_block_8/multi_head_attention_8/dense_99/Tensordot/ReadVariableOpB^transformer_block_8/sequential_8/dense_100/BiasAdd/ReadVariableOpD^transformer_block_8/sequential_8/dense_100/Tensordot/ReadVariableOpB^transformer_block_8/sequential_8/dense_101/BiasAdd/ReadVariableOpD^transformer_block_8/sequential_8/dense_101/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????d: : : : : : : : : : : : : : : : : : : : : : 2D
 dense_102/BiasAdd/ReadVariableOp dense_102/BiasAdd/ReadVariableOp2B
dense_102/MatMul/ReadVariableOpdense_102/MatMul/ReadVariableOp2D
 dense_103/BiasAdd/ReadVariableOp dense_103/BiasAdd/ReadVariableOp2B
dense_103/MatMul/ReadVariableOpdense_103/MatMul/ReadVariableOp2|
<token_and_position_embedding_9/embedding_20/embedding_lookup<token_and_position_embedding_9/embedding_20/embedding_lookup2|
<token_and_position_embedding_9/embedding_21/embedding_lookup<token_and_position_embedding_9/embedding_21/embedding_lookup2?
Ctransformer_block_8/layer_normalization_26/batchnorm/ReadVariableOpCtransformer_block_8/layer_normalization_26/batchnorm/ReadVariableOp2?
Gtransformer_block_8/layer_normalization_26/batchnorm/mul/ReadVariableOpGtransformer_block_8/layer_normalization_26/batchnorm/mul/ReadVariableOp2?
Ctransformer_block_8/layer_normalization_27/batchnorm/ReadVariableOpCtransformer_block_8/layer_normalization_27/batchnorm/ReadVariableOp2?
Gtransformer_block_8/layer_normalization_27/batchnorm/mul/ReadVariableOpGtransformer_block_8/layer_normalization_27/batchnorm/mul/ReadVariableOp2?
Jtransformer_block_8/multi_head_attention_8/dense_96/BiasAdd/ReadVariableOpJtransformer_block_8/multi_head_attention_8/dense_96/BiasAdd/ReadVariableOp2?
Ltransformer_block_8/multi_head_attention_8/dense_96/Tensordot/ReadVariableOpLtransformer_block_8/multi_head_attention_8/dense_96/Tensordot/ReadVariableOp2?
Jtransformer_block_8/multi_head_attention_8/dense_97/BiasAdd/ReadVariableOpJtransformer_block_8/multi_head_attention_8/dense_97/BiasAdd/ReadVariableOp2?
Ltransformer_block_8/multi_head_attention_8/dense_97/Tensordot/ReadVariableOpLtransformer_block_8/multi_head_attention_8/dense_97/Tensordot/ReadVariableOp2?
Jtransformer_block_8/multi_head_attention_8/dense_98/BiasAdd/ReadVariableOpJtransformer_block_8/multi_head_attention_8/dense_98/BiasAdd/ReadVariableOp2?
Ltransformer_block_8/multi_head_attention_8/dense_98/Tensordot/ReadVariableOpLtransformer_block_8/multi_head_attention_8/dense_98/Tensordot/ReadVariableOp2?
Jtransformer_block_8/multi_head_attention_8/dense_99/BiasAdd/ReadVariableOpJtransformer_block_8/multi_head_attention_8/dense_99/BiasAdd/ReadVariableOp2?
Ltransformer_block_8/multi_head_attention_8/dense_99/Tensordot/ReadVariableOpLtransformer_block_8/multi_head_attention_8/dense_99/Tensordot/ReadVariableOp2?
Atransformer_block_8/sequential_8/dense_100/BiasAdd/ReadVariableOpAtransformer_block_8/sequential_8/dense_100/BiasAdd/ReadVariableOp2?
Ctransformer_block_8/sequential_8/dense_100/Tensordot/ReadVariableOpCtransformer_block_8/sequential_8/dense_100/Tensordot/ReadVariableOp2?
Atransformer_block_8/sequential_8/dense_101/BiasAdd/ReadVariableOpAtransformer_block_8/sequential_8/dense_101/BiasAdd/ReadVariableOp2?
Ctransformer_block_8/sequential_8/dense_101/Tensordot/ReadVariableOpCtransformer_block_8/sequential_8/dense_101/Tensordot/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?+
?	
B__inference_model_8_layer_call_and_return_conditional_losses_91399

inputs6
$token_and_position_embedding_9_91060:d@7
$token_and_position_embedding_9_91062:	?@@+
transformer_block_8_91310:@@'
transformer_block_8_91312:@+
transformer_block_8_91314:@@'
transformer_block_8_91316:@+
transformer_block_8_91318:@@'
transformer_block_8_91320:@+
transformer_block_8_91322:@@'
transformer_block_8_91324:@'
transformer_block_8_91326:@'
transformer_block_8_91328:@+
transformer_block_8_91330:@ '
transformer_block_8_91332: +
transformer_block_8_91334: @'
transformer_block_8_91336:@'
transformer_block_8_91338:@'
transformer_block_8_91340:@!
dense_102_91369:@
dense_102_91371:!
dense_103_91393:
dense_103_91395:
identity??!dense_102/StatefulPartitionedCall?!dense_103/StatefulPartitionedCall?6token_and_position_embedding_9/StatefulPartitionedCall?+transformer_block_8/StatefulPartitionedCall?
6token_and_position_embedding_9/StatefulPartitionedCallStatefulPartitionedCallinputs$token_and_position_embedding_9_91060$token_and_position_embedding_9_91062*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *b
f]R[
Y__inference_token_and_position_embedding_9_layer_call_and_return_conditional_losses_9105928
6token_and_position_embedding_9/StatefulPartitionedCall?
+transformer_block_8/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_9/StatefulPartitionedCall:output:0transformer_block_8_91310transformer_block_8_91312transformer_block_8_91314transformer_block_8_91316transformer_block_8_91318transformer_block_8_91320transformer_block_8_91322transformer_block_8_91324transformer_block_8_91326transformer_block_8_91328transformer_block_8_91330transformer_block_8_91332transformer_block_8_91334transformer_block_8_91336transformer_block_8_91338transformer_block_8_91340*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d@*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_transformer_block_8_layer_call_and_return_conditional_losses_913092-
+transformer_block_8/StatefulPartitionedCall?
*global_average_pooling1d_8/PartitionedCallPartitionedCall4transformer_block_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_913482,
*global_average_pooling1d_8/PartitionedCall?
dropout_44/PartitionedCallPartitionedCall3global_average_pooling1d_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_44_layer_call_and_return_conditional_losses_913552
dropout_44/PartitionedCall?
!dense_102/StatefulPartitionedCallStatefulPartitionedCall#dropout_44/PartitionedCall:output:0dense_102_91369dense_102_91371*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_102_layer_call_and_return_conditional_losses_913682#
!dense_102/StatefulPartitionedCall?
dropout_45/PartitionedCallPartitionedCall*dense_102/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_45_layer_call_and_return_conditional_losses_913792
dropout_45/PartitionedCall?
!dense_103/StatefulPartitionedCallStatefulPartitionedCall#dropout_45/PartitionedCall:output:0dense_103_91393dense_103_91395*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_103_layer_call_and_return_conditional_losses_913922#
!dense_103/StatefulPartitionedCall?
IdentityIdentity*dense_103/StatefulPartitionedCall:output:0"^dense_102/StatefulPartitionedCall"^dense_103/StatefulPartitionedCall7^token_and_position_embedding_9/StatefulPartitionedCall,^transformer_block_8/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????d: : : : : : : : : : : : : : : : : : : : : : 2F
!dense_102/StatefulPartitionedCall!dense_102/StatefulPartitionedCall2F
!dense_103/StatefulPartitionedCall!dense_103/StatefulPartitionedCall2p
6token_and_position_embedding_9/StatefulPartitionedCall6token_and_position_embedding_9/StatefulPartitionedCall2Z
+transformer_block_8/StatefulPartitionedCall+transformer_block_8/StatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
'__inference_model_8_layer_call_fn_91446
input_11
unknown:d@
	unknown_0:	?@@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@ 

unknown_12: 

unknown_13: @

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:

unknown_19:

unknown_20:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_11unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_8_layer_call_and_return_conditional_losses_913992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????d: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????d
"
_user_specified_name
input_11
?

?
D__inference_dense_102_layer_call_and_return_conditional_losses_93597

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
c
E__inference_dropout_45_layer_call_and_return_conditional_losses_93612

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
3__inference_transformer_block_8_layer_call_fn_92989

inputs
unknown:@@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@ 

unknown_10: 

unknown_11: @

unknown_12:@

unknown_13:@

unknown_14:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d@*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_transformer_block_8_layer_call_and_return_conditional_losses_913092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????d@: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????d@
 
_user_specified_nameinputs
?
V
:__inference_global_average_pooling1d_8_layer_call_fn_93538

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_913482
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d@:S O
+
_output_shapes
:?????????d@
 
_user_specified_nameinputs
?
?
,__inference_sequential_8_layer_call_fn_90903
dense_100_input
unknown:@ 
	unknown_0: 
	unknown_1: @
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_100_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_8_layer_call_and_return_conditional_losses_908922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????d@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:?????????d@
)
_user_specified_namedense_100_input
?
c
E__inference_dropout_45_layer_call_and_return_conditional_losses_91379

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?L
?
G__inference_sequential_8_layer_call_and_return_conditional_losses_93784

inputs=
+dense_100_tensordot_readvariableop_resource:@ 7
)dense_100_biasadd_readvariableop_resource: =
+dense_101_tensordot_readvariableop_resource: @7
)dense_101_biasadd_readvariableop_resource:@
identity?? dense_100/BiasAdd/ReadVariableOp?"dense_100/Tensordot/ReadVariableOp? dense_101/BiasAdd/ReadVariableOp?"dense_101/Tensordot/ReadVariableOp?
"dense_100/Tensordot/ReadVariableOpReadVariableOp+dense_100_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02$
"dense_100/Tensordot/ReadVariableOp~
dense_100/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_100/Tensordot/axes?
dense_100/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_100/Tensordot/freel
dense_100/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_100/Tensordot/Shape?
!dense_100/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_100/Tensordot/GatherV2/axis?
dense_100/Tensordot/GatherV2GatherV2"dense_100/Tensordot/Shape:output:0!dense_100/Tensordot/free:output:0*dense_100/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_100/Tensordot/GatherV2?
#dense_100/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_100/Tensordot/GatherV2_1/axis?
dense_100/Tensordot/GatherV2_1GatherV2"dense_100/Tensordot/Shape:output:0!dense_100/Tensordot/axes:output:0,dense_100/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_100/Tensordot/GatherV2_1?
dense_100/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_100/Tensordot/Const?
dense_100/Tensordot/ProdProd%dense_100/Tensordot/GatherV2:output:0"dense_100/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_100/Tensordot/Prod?
dense_100/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_100/Tensordot/Const_1?
dense_100/Tensordot/Prod_1Prod'dense_100/Tensordot/GatherV2_1:output:0$dense_100/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_100/Tensordot/Prod_1?
dense_100/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_100/Tensordot/concat/axis?
dense_100/Tensordot/concatConcatV2!dense_100/Tensordot/free:output:0!dense_100/Tensordot/axes:output:0(dense_100/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_100/Tensordot/concat?
dense_100/Tensordot/stackPack!dense_100/Tensordot/Prod:output:0#dense_100/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_100/Tensordot/stack?
dense_100/Tensordot/transpose	Transposeinputs#dense_100/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????d@2
dense_100/Tensordot/transpose?
dense_100/Tensordot/ReshapeReshape!dense_100/Tensordot/transpose:y:0"dense_100/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_100/Tensordot/Reshape?
dense_100/Tensordot/MatMulMatMul$dense_100/Tensordot/Reshape:output:0*dense_100/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_100/Tensordot/MatMul?
dense_100/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_100/Tensordot/Const_2?
!dense_100/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_100/Tensordot/concat_1/axis?
dense_100/Tensordot/concat_1ConcatV2%dense_100/Tensordot/GatherV2:output:0$dense_100/Tensordot/Const_2:output:0*dense_100/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_100/Tensordot/concat_1?
dense_100/TensordotReshape$dense_100/Tensordot/MatMul:product:0%dense_100/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????d 2
dense_100/Tensordot?
 dense_100/BiasAdd/ReadVariableOpReadVariableOp)dense_100_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 dense_100/BiasAdd/ReadVariableOp?
dense_100/BiasAddBiasAdddense_100/Tensordot:output:0(dense_100/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d 2
dense_100/BiasAddz
dense_100/ReluReludense_100/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d 2
dense_100/Relu?
"dense_101/Tensordot/ReadVariableOpReadVariableOp+dense_101_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02$
"dense_101/Tensordot/ReadVariableOp~
dense_101/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_101/Tensordot/axes?
dense_101/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_101/Tensordot/free?
dense_101/Tensordot/ShapeShapedense_100/Relu:activations:0*
T0*
_output_shapes
:2
dense_101/Tensordot/Shape?
!dense_101/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_101/Tensordot/GatherV2/axis?
dense_101/Tensordot/GatherV2GatherV2"dense_101/Tensordot/Shape:output:0!dense_101/Tensordot/free:output:0*dense_101/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_101/Tensordot/GatherV2?
#dense_101/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_101/Tensordot/GatherV2_1/axis?
dense_101/Tensordot/GatherV2_1GatherV2"dense_101/Tensordot/Shape:output:0!dense_101/Tensordot/axes:output:0,dense_101/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_101/Tensordot/GatherV2_1?
dense_101/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_101/Tensordot/Const?
dense_101/Tensordot/ProdProd%dense_101/Tensordot/GatherV2:output:0"dense_101/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_101/Tensordot/Prod?
dense_101/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_101/Tensordot/Const_1?
dense_101/Tensordot/Prod_1Prod'dense_101/Tensordot/GatherV2_1:output:0$dense_101/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_101/Tensordot/Prod_1?
dense_101/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_101/Tensordot/concat/axis?
dense_101/Tensordot/concatConcatV2!dense_101/Tensordot/free:output:0!dense_101/Tensordot/axes:output:0(dense_101/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_101/Tensordot/concat?
dense_101/Tensordot/stackPack!dense_101/Tensordot/Prod:output:0#dense_101/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_101/Tensordot/stack?
dense_101/Tensordot/transpose	Transposedense_100/Relu:activations:0#dense_101/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????d 2
dense_101/Tensordot/transpose?
dense_101/Tensordot/ReshapeReshape!dense_101/Tensordot/transpose:y:0"dense_101/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_101/Tensordot/Reshape?
dense_101/Tensordot/MatMulMatMul$dense_101/Tensordot/Reshape:output:0*dense_101/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_101/Tensordot/MatMul?
dense_101/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_101/Tensordot/Const_2?
!dense_101/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_101/Tensordot/concat_1/axis?
dense_101/Tensordot/concat_1ConcatV2%dense_101/Tensordot/GatherV2:output:0$dense_101/Tensordot/Const_2:output:0*dense_101/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_101/Tensordot/concat_1?
dense_101/TensordotReshape$dense_101/Tensordot/MatMul:product:0%dense_101/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????d@2
dense_101/Tensordot?
 dense_101/BiasAdd/ReadVariableOpReadVariableOp)dense_101_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 dense_101/BiasAdd/ReadVariableOp?
dense_101/BiasAddBiasAdddense_101/Tensordot:output:0(dense_101/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@2
dense_101/BiasAdd?
IdentityIdentitydense_101/BiasAdd:output:0!^dense_100/BiasAdd/ReadVariableOp#^dense_100/Tensordot/ReadVariableOp!^dense_101/BiasAdd/ReadVariableOp#^dense_101/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????d@: : : : 2D
 dense_100/BiasAdd/ReadVariableOp dense_100/BiasAdd/ReadVariableOp2H
"dense_100/Tensordot/ReadVariableOp"dense_100/Tensordot/ReadVariableOp2D
 dense_101/BiasAdd/ReadVariableOp dense_101/BiasAdd/ReadVariableOp2H
"dense_101/Tensordot/ReadVariableOp"dense_101/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????d@
 
_user_specified_nameinputs
?
c
E__inference_dropout_44_layer_call_and_return_conditional_losses_93565

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????@2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
d
E__inference_dropout_45_layer_call_and_return_conditional_losses_91476

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_sequential_8_layer_call_and_return_conditional_losses_90892

inputs!
dense_100_90850:@ 
dense_100_90852: !
dense_101_90886: @
dense_101_90888:@
identity??!dense_100/StatefulPartitionedCall?!dense_101/StatefulPartitionedCall?
!dense_100/StatefulPartitionedCallStatefulPartitionedCallinputsdense_100_90850dense_100_90852*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_100_layer_call_and_return_conditional_losses_908492#
!dense_100/StatefulPartitionedCall?
!dense_101/StatefulPartitionedCallStatefulPartitionedCall*dense_100/StatefulPartitionedCall:output:0dense_101_90886dense_101_90888*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_101_layer_call_and_return_conditional_losses_908852#
!dense_101/StatefulPartitionedCall?
IdentityIdentity*dense_101/StatefulPartitionedCall:output:0"^dense_100/StatefulPartitionedCall"^dense_101/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????d@: : : : 2F
!dense_100/StatefulPartitionedCall!dense_100/StatefulPartitionedCall2F
!dense_101/StatefulPartitionedCall!dense_101/StatefulPartitionedCall:S O
+
_output_shapes
:?????????d@
 
_user_specified_nameinputs
??
?+
__inference__traced_save_94111
file_prefix/
+savev2_dense_102_kernel_read_readvariableop-
)savev2_dense_102_bias_read_readvariableop/
+savev2_dense_103_kernel_read_readvariableop-
)savev2_dense_103_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopU
Qsavev2_token_and_position_embedding_9_embedding_20_embeddings_read_readvariableopU
Qsavev2_token_and_position_embedding_9_embedding_21_embeddings_read_readvariableopY
Usavev2_transformer_block_8_multi_head_attention_8_dense_96_kernel_read_readvariableopW
Ssavev2_transformer_block_8_multi_head_attention_8_dense_96_bias_read_readvariableopY
Usavev2_transformer_block_8_multi_head_attention_8_dense_97_kernel_read_readvariableopW
Ssavev2_transformer_block_8_multi_head_attention_8_dense_97_bias_read_readvariableopY
Usavev2_transformer_block_8_multi_head_attention_8_dense_98_kernel_read_readvariableopW
Ssavev2_transformer_block_8_multi_head_attention_8_dense_98_bias_read_readvariableopY
Usavev2_transformer_block_8_multi_head_attention_8_dense_99_kernel_read_readvariableopW
Ssavev2_transformer_block_8_multi_head_attention_8_dense_99_bias_read_readvariableop/
+savev2_dense_100_kernel_read_readvariableop-
)savev2_dense_100_bias_read_readvariableop/
+savev2_dense_101_kernel_read_readvariableop-
)savev2_dense_101_bias_read_readvariableopO
Ksavev2_transformer_block_8_layer_normalization_26_gamma_read_readvariableopN
Jsavev2_transformer_block_8_layer_normalization_26_beta_read_readvariableopO
Ksavev2_transformer_block_8_layer_normalization_27_gamma_read_readvariableopN
Jsavev2_transformer_block_8_layer_normalization_27_beta_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_dense_102_kernel_m_read_readvariableop4
0savev2_adam_dense_102_bias_m_read_readvariableop6
2savev2_adam_dense_103_kernel_m_read_readvariableop4
0savev2_adam_dense_103_bias_m_read_readvariableop\
Xsavev2_adam_token_and_position_embedding_9_embedding_20_embeddings_m_read_readvariableop\
Xsavev2_adam_token_and_position_embedding_9_embedding_21_embeddings_m_read_readvariableop`
\savev2_adam_transformer_block_8_multi_head_attention_8_dense_96_kernel_m_read_readvariableop^
Zsavev2_adam_transformer_block_8_multi_head_attention_8_dense_96_bias_m_read_readvariableop`
\savev2_adam_transformer_block_8_multi_head_attention_8_dense_97_kernel_m_read_readvariableop^
Zsavev2_adam_transformer_block_8_multi_head_attention_8_dense_97_bias_m_read_readvariableop`
\savev2_adam_transformer_block_8_multi_head_attention_8_dense_98_kernel_m_read_readvariableop^
Zsavev2_adam_transformer_block_8_multi_head_attention_8_dense_98_bias_m_read_readvariableop`
\savev2_adam_transformer_block_8_multi_head_attention_8_dense_99_kernel_m_read_readvariableop^
Zsavev2_adam_transformer_block_8_multi_head_attention_8_dense_99_bias_m_read_readvariableop6
2savev2_adam_dense_100_kernel_m_read_readvariableop4
0savev2_adam_dense_100_bias_m_read_readvariableop6
2savev2_adam_dense_101_kernel_m_read_readvariableop4
0savev2_adam_dense_101_bias_m_read_readvariableopV
Rsavev2_adam_transformer_block_8_layer_normalization_26_gamma_m_read_readvariableopU
Qsavev2_adam_transformer_block_8_layer_normalization_26_beta_m_read_readvariableopV
Rsavev2_adam_transformer_block_8_layer_normalization_27_gamma_m_read_readvariableopU
Qsavev2_adam_transformer_block_8_layer_normalization_27_beta_m_read_readvariableop6
2savev2_adam_dense_102_kernel_v_read_readvariableop4
0savev2_adam_dense_102_bias_v_read_readvariableop6
2savev2_adam_dense_103_kernel_v_read_readvariableop4
0savev2_adam_dense_103_bias_v_read_readvariableop\
Xsavev2_adam_token_and_position_embedding_9_embedding_20_embeddings_v_read_readvariableop\
Xsavev2_adam_token_and_position_embedding_9_embedding_21_embeddings_v_read_readvariableop`
\savev2_adam_transformer_block_8_multi_head_attention_8_dense_96_kernel_v_read_readvariableop^
Zsavev2_adam_transformer_block_8_multi_head_attention_8_dense_96_bias_v_read_readvariableop`
\savev2_adam_transformer_block_8_multi_head_attention_8_dense_97_kernel_v_read_readvariableop^
Zsavev2_adam_transformer_block_8_multi_head_attention_8_dense_97_bias_v_read_readvariableop`
\savev2_adam_transformer_block_8_multi_head_attention_8_dense_98_kernel_v_read_readvariableop^
Zsavev2_adam_transformer_block_8_multi_head_attention_8_dense_98_bias_v_read_readvariableop`
\savev2_adam_transformer_block_8_multi_head_attention_8_dense_99_kernel_v_read_readvariableop^
Zsavev2_adam_transformer_block_8_multi_head_attention_8_dense_99_bias_v_read_readvariableop6
2savev2_adam_dense_100_kernel_v_read_readvariableop4
0savev2_adam_dense_100_bias_v_read_readvariableop6
2savev2_adam_dense_101_kernel_v_read_readvariableop4
0savev2_adam_dense_101_bias_v_read_readvariableopV
Rsavev2_adam_transformer_block_8_layer_normalization_26_gamma_v_read_readvariableopU
Qsavev2_adam_transformer_block_8_layer_normalization_26_beta_v_read_readvariableopV
Rsavev2_adam_transformer_block_8_layer_normalization_27_gamma_v_read_readvariableopU
Qsavev2_adam_transformer_block_8_layer_normalization_27_beta_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?(
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*?'
value?'B?'LB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*?
value?B?LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?*
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_102_kernel_read_readvariableop)savev2_dense_102_bias_read_readvariableop+savev2_dense_103_kernel_read_readvariableop)savev2_dense_103_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopQsavev2_token_and_position_embedding_9_embedding_20_embeddings_read_readvariableopQsavev2_token_and_position_embedding_9_embedding_21_embeddings_read_readvariableopUsavev2_transformer_block_8_multi_head_attention_8_dense_96_kernel_read_readvariableopSsavev2_transformer_block_8_multi_head_attention_8_dense_96_bias_read_readvariableopUsavev2_transformer_block_8_multi_head_attention_8_dense_97_kernel_read_readvariableopSsavev2_transformer_block_8_multi_head_attention_8_dense_97_bias_read_readvariableopUsavev2_transformer_block_8_multi_head_attention_8_dense_98_kernel_read_readvariableopSsavev2_transformer_block_8_multi_head_attention_8_dense_98_bias_read_readvariableopUsavev2_transformer_block_8_multi_head_attention_8_dense_99_kernel_read_readvariableopSsavev2_transformer_block_8_multi_head_attention_8_dense_99_bias_read_readvariableop+savev2_dense_100_kernel_read_readvariableop)savev2_dense_100_bias_read_readvariableop+savev2_dense_101_kernel_read_readvariableop)savev2_dense_101_bias_read_readvariableopKsavev2_transformer_block_8_layer_normalization_26_gamma_read_readvariableopJsavev2_transformer_block_8_layer_normalization_26_beta_read_readvariableopKsavev2_transformer_block_8_layer_normalization_27_gamma_read_readvariableopJsavev2_transformer_block_8_layer_normalization_27_beta_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_dense_102_kernel_m_read_readvariableop0savev2_adam_dense_102_bias_m_read_readvariableop2savev2_adam_dense_103_kernel_m_read_readvariableop0savev2_adam_dense_103_bias_m_read_readvariableopXsavev2_adam_token_and_position_embedding_9_embedding_20_embeddings_m_read_readvariableopXsavev2_adam_token_and_position_embedding_9_embedding_21_embeddings_m_read_readvariableop\savev2_adam_transformer_block_8_multi_head_attention_8_dense_96_kernel_m_read_readvariableopZsavev2_adam_transformer_block_8_multi_head_attention_8_dense_96_bias_m_read_readvariableop\savev2_adam_transformer_block_8_multi_head_attention_8_dense_97_kernel_m_read_readvariableopZsavev2_adam_transformer_block_8_multi_head_attention_8_dense_97_bias_m_read_readvariableop\savev2_adam_transformer_block_8_multi_head_attention_8_dense_98_kernel_m_read_readvariableopZsavev2_adam_transformer_block_8_multi_head_attention_8_dense_98_bias_m_read_readvariableop\savev2_adam_transformer_block_8_multi_head_attention_8_dense_99_kernel_m_read_readvariableopZsavev2_adam_transformer_block_8_multi_head_attention_8_dense_99_bias_m_read_readvariableop2savev2_adam_dense_100_kernel_m_read_readvariableop0savev2_adam_dense_100_bias_m_read_readvariableop2savev2_adam_dense_101_kernel_m_read_readvariableop0savev2_adam_dense_101_bias_m_read_readvariableopRsavev2_adam_transformer_block_8_layer_normalization_26_gamma_m_read_readvariableopQsavev2_adam_transformer_block_8_layer_normalization_26_beta_m_read_readvariableopRsavev2_adam_transformer_block_8_layer_normalization_27_gamma_m_read_readvariableopQsavev2_adam_transformer_block_8_layer_normalization_27_beta_m_read_readvariableop2savev2_adam_dense_102_kernel_v_read_readvariableop0savev2_adam_dense_102_bias_v_read_readvariableop2savev2_adam_dense_103_kernel_v_read_readvariableop0savev2_adam_dense_103_bias_v_read_readvariableopXsavev2_adam_token_and_position_embedding_9_embedding_20_embeddings_v_read_readvariableopXsavev2_adam_token_and_position_embedding_9_embedding_21_embeddings_v_read_readvariableop\savev2_adam_transformer_block_8_multi_head_attention_8_dense_96_kernel_v_read_readvariableopZsavev2_adam_transformer_block_8_multi_head_attention_8_dense_96_bias_v_read_readvariableop\savev2_adam_transformer_block_8_multi_head_attention_8_dense_97_kernel_v_read_readvariableopZsavev2_adam_transformer_block_8_multi_head_attention_8_dense_97_bias_v_read_readvariableop\savev2_adam_transformer_block_8_multi_head_attention_8_dense_98_kernel_v_read_readvariableopZsavev2_adam_transformer_block_8_multi_head_attention_8_dense_98_bias_v_read_readvariableop\savev2_adam_transformer_block_8_multi_head_attention_8_dense_99_kernel_v_read_readvariableopZsavev2_adam_transformer_block_8_multi_head_attention_8_dense_99_bias_v_read_readvariableop2savev2_adam_dense_100_kernel_v_read_readvariableop0savev2_adam_dense_100_bias_v_read_readvariableop2savev2_adam_dense_101_kernel_v_read_readvariableop0savev2_adam_dense_101_bias_v_read_readvariableopRsavev2_adam_transformer_block_8_layer_normalization_26_gamma_v_read_readvariableopQsavev2_adam_transformer_block_8_layer_normalization_26_beta_v_read_readvariableopRsavev2_adam_transformer_block_8_layer_normalization_27_gamma_v_read_readvariableopQsavev2_adam_transformer_block_8_layer_normalization_27_beta_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *Z
dtypesP
N2L	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :@:::: : : : : :	?@@:d@:@@:@:@@:@:@@:@:@@:@:@ : : @:@:@:@:@:@: : : : :@::::	?@@:d@:@@:@:@@:@:@@:@:@@:@:@ : : @:@:@:@:@:@:@::::	?@@:d@:@@:@:@@:@:@@:@:@@:@:@ : : @:@:@:@:@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :%
!

_output_shapes
:	?@@:$ 

_output_shapes

:d@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: @: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$  

_output_shapes

:@: !

_output_shapes
::$" 

_output_shapes

:: #

_output_shapes
::%$!

_output_shapes
:	?@@:$% 

_output_shapes

:d@:$& 

_output_shapes

:@@: '

_output_shapes
:@:$( 

_output_shapes

:@@: )

_output_shapes
:@:$* 

_output_shapes

:@@: +

_output_shapes
:@:$, 

_output_shapes

:@@: -

_output_shapes
:@:$. 

_output_shapes

:@ : /

_output_shapes
: :$0 

_output_shapes

: @: 1

_output_shapes
:@: 2

_output_shapes
:@: 3

_output_shapes
:@: 4

_output_shapes
:@: 5

_output_shapes
:@:$6 

_output_shapes

:@: 7

_output_shapes
::$8 

_output_shapes

:: 9

_output_shapes
::%:!

_output_shapes
:	?@@:$; 

_output_shapes

:d@:$< 

_output_shapes

:@@: =

_output_shapes
:@:$> 

_output_shapes

:@@: ?

_output_shapes
:@:$@ 

_output_shapes

:@@: A

_output_shapes
:@:$B 

_output_shapes

:@@: C

_output_shapes
:@:$D 

_output_shapes

:@ : E

_output_shapes
: :$F 

_output_shapes

: @: G

_output_shapes
:@: H

_output_shapes
:@: I

_output_shapes
:@: J

_output_shapes
:@: K

_output_shapes
:@:L

_output_shapes
: 
?
?
,__inference_sequential_8_layer_call_fn_90976
dense_100_input
unknown:@ 
	unknown_0: 
	unknown_1: @
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_100_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_8_layer_call_and_return_conditional_losses_909522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????d@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:?????????d@
)
_user_specified_namedense_100_input
?
?
'__inference_model_8_layer_call_fn_92062
input_11
unknown:d@
	unknown_0:	?@@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@ 

unknown_12: 

unknown_13: @

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:

unknown_19:

unknown_20:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_11unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_8_layer_call_and_return_conditional_losses_919662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????d: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????d
"
_user_specified_name
input_11
?
q
U__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_93550

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indiceso
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@2
Meana
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d@:S O
+
_output_shapes
:?????????d@
 
_user_specified_nameinputs
?
?
)__inference_dense_101_layer_call_fn_93833

inputs
unknown: @
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_101_layer_call_and_return_conditional_losses_908852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????d : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????d 
 
_user_specified_nameinputs
?
d
E__inference_dropout_44_layer_call_and_return_conditional_losses_93577

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
c
*__inference_dropout_44_layer_call_fn_93560

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_44_layer_call_and_return_conditional_losses_915092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
d
E__inference_dropout_45_layer_call_and_return_conditional_losses_93624

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
F
*__inference_dropout_44_layer_call_fn_93555

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_44_layer_call_and_return_conditional_losses_913552
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
'__inference_model_8_layer_call_fn_92327

inputs
unknown:d@
	unknown_0:	?@@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@ 

unknown_12: 

unknown_13: @

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:

unknown_19:

unknown_20:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_model_8_layer_call_and_return_conditional_losses_919662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????d: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_92229
input_11
unknown:d@
	unknown_0:	?@@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@

unknown_10:@

unknown_11:@ 

unknown_12: 

unknown_13: @

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:

unknown_19:

unknown_20:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_11unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_908112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????d: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????d
"
_user_specified_name
input_11
?/
?	
B__inference_model_8_layer_call_and_return_conditional_losses_92172
input_116
$token_and_position_embedding_9_92120:d@7
$token_and_position_embedding_9_92122:	?@@+
transformer_block_8_92125:@@'
transformer_block_8_92127:@+
transformer_block_8_92129:@@'
transformer_block_8_92131:@+
transformer_block_8_92133:@@'
transformer_block_8_92135:@+
transformer_block_8_92137:@@'
transformer_block_8_92139:@'
transformer_block_8_92141:@'
transformer_block_8_92143:@+
transformer_block_8_92145:@ '
transformer_block_8_92147: +
transformer_block_8_92149: @'
transformer_block_8_92151:@'
transformer_block_8_92153:@'
transformer_block_8_92155:@!
dense_102_92160:@
dense_102_92162:!
dense_103_92166:
dense_103_92168:
identity??!dense_102/StatefulPartitionedCall?!dense_103/StatefulPartitionedCall?"dropout_44/StatefulPartitionedCall?"dropout_45/StatefulPartitionedCall?6token_and_position_embedding_9/StatefulPartitionedCall?+transformer_block_8/StatefulPartitionedCall?
6token_and_position_embedding_9/StatefulPartitionedCallStatefulPartitionedCallinput_11$token_and_position_embedding_9_92120$token_and_position_embedding_9_92122*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *b
f]R[
Y__inference_token_and_position_embedding_9_layer_call_and_return_conditional_losses_9105928
6token_and_position_embedding_9/StatefulPartitionedCall?
+transformer_block_8/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_9/StatefulPartitionedCall:output:0transformer_block_8_92125transformer_block_8_92127transformer_block_8_92129transformer_block_8_92131transformer_block_8_92133transformer_block_8_92135transformer_block_8_92137transformer_block_8_92139transformer_block_8_92141transformer_block_8_92143transformer_block_8_92145transformer_block_8_92147transformer_block_8_92149transformer_block_8_92151transformer_block_8_92153transformer_block_8_92155*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d@*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_transformer_block_8_layer_call_and_return_conditional_losses_918152-
+transformer_block_8/StatefulPartitionedCall?
*global_average_pooling1d_8/PartitionedCallPartitionedCall4transformer_block_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_913482,
*global_average_pooling1d_8/PartitionedCall?
"dropout_44/StatefulPartitionedCallStatefulPartitionedCall3global_average_pooling1d_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_44_layer_call_and_return_conditional_losses_915092$
"dropout_44/StatefulPartitionedCall?
!dense_102/StatefulPartitionedCallStatefulPartitionedCall+dropout_44/StatefulPartitionedCall:output:0dense_102_92160dense_102_92162*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_102_layer_call_and_return_conditional_losses_913682#
!dense_102/StatefulPartitionedCall?
"dropout_45/StatefulPartitionedCallStatefulPartitionedCall*dense_102/StatefulPartitionedCall:output:0#^dropout_44/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_45_layer_call_and_return_conditional_losses_914762$
"dropout_45/StatefulPartitionedCall?
!dense_103/StatefulPartitionedCallStatefulPartitionedCall+dropout_45/StatefulPartitionedCall:output:0dense_103_92166dense_103_92168*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_103_layer_call_and_return_conditional_losses_913922#
!dense_103/StatefulPartitionedCall?
IdentityIdentity*dense_103/StatefulPartitionedCall:output:0"^dense_102/StatefulPartitionedCall"^dense_103/StatefulPartitionedCall#^dropout_44/StatefulPartitionedCall#^dropout_45/StatefulPartitionedCall7^token_and_position_embedding_9/StatefulPartitionedCall,^transformer_block_8/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????d: : : : : : : : : : : : : : : : : : : : : : 2F
!dense_102/StatefulPartitionedCall!dense_102/StatefulPartitionedCall2F
!dense_103/StatefulPartitionedCall!dense_103/StatefulPartitionedCall2H
"dropout_44/StatefulPartitionedCall"dropout_44/StatefulPartitionedCall2H
"dropout_45/StatefulPartitionedCall"dropout_45/StatefulPartitionedCall2p
6token_and_position_embedding_9/StatefulPartitionedCall6token_and_position_embedding_9/StatefulPartitionedCall2Z
+transformer_block_8/StatefulPartitionedCall+transformer_block_8/StatefulPartitionedCall:Q M
'
_output_shapes
:?????????d
"
_user_specified_name
input_11
?L
?
G__inference_sequential_8_layer_call_and_return_conditional_losses_93727

inputs=
+dense_100_tensordot_readvariableop_resource:@ 7
)dense_100_biasadd_readvariableop_resource: =
+dense_101_tensordot_readvariableop_resource: @7
)dense_101_biasadd_readvariableop_resource:@
identity?? dense_100/BiasAdd/ReadVariableOp?"dense_100/Tensordot/ReadVariableOp? dense_101/BiasAdd/ReadVariableOp?"dense_101/Tensordot/ReadVariableOp?
"dense_100/Tensordot/ReadVariableOpReadVariableOp+dense_100_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02$
"dense_100/Tensordot/ReadVariableOp~
dense_100/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_100/Tensordot/axes?
dense_100/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_100/Tensordot/freel
dense_100/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_100/Tensordot/Shape?
!dense_100/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_100/Tensordot/GatherV2/axis?
dense_100/Tensordot/GatherV2GatherV2"dense_100/Tensordot/Shape:output:0!dense_100/Tensordot/free:output:0*dense_100/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_100/Tensordot/GatherV2?
#dense_100/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_100/Tensordot/GatherV2_1/axis?
dense_100/Tensordot/GatherV2_1GatherV2"dense_100/Tensordot/Shape:output:0!dense_100/Tensordot/axes:output:0,dense_100/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_100/Tensordot/GatherV2_1?
dense_100/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_100/Tensordot/Const?
dense_100/Tensordot/ProdProd%dense_100/Tensordot/GatherV2:output:0"dense_100/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_100/Tensordot/Prod?
dense_100/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_100/Tensordot/Const_1?
dense_100/Tensordot/Prod_1Prod'dense_100/Tensordot/GatherV2_1:output:0$dense_100/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_100/Tensordot/Prod_1?
dense_100/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_100/Tensordot/concat/axis?
dense_100/Tensordot/concatConcatV2!dense_100/Tensordot/free:output:0!dense_100/Tensordot/axes:output:0(dense_100/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_100/Tensordot/concat?
dense_100/Tensordot/stackPack!dense_100/Tensordot/Prod:output:0#dense_100/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_100/Tensordot/stack?
dense_100/Tensordot/transpose	Transposeinputs#dense_100/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????d@2
dense_100/Tensordot/transpose?
dense_100/Tensordot/ReshapeReshape!dense_100/Tensordot/transpose:y:0"dense_100/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_100/Tensordot/Reshape?
dense_100/Tensordot/MatMulMatMul$dense_100/Tensordot/Reshape:output:0*dense_100/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_100/Tensordot/MatMul?
dense_100/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_100/Tensordot/Const_2?
!dense_100/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_100/Tensordot/concat_1/axis?
dense_100/Tensordot/concat_1ConcatV2%dense_100/Tensordot/GatherV2:output:0$dense_100/Tensordot/Const_2:output:0*dense_100/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_100/Tensordot/concat_1?
dense_100/TensordotReshape$dense_100/Tensordot/MatMul:product:0%dense_100/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????d 2
dense_100/Tensordot?
 dense_100/BiasAdd/ReadVariableOpReadVariableOp)dense_100_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 dense_100/BiasAdd/ReadVariableOp?
dense_100/BiasAddBiasAdddense_100/Tensordot:output:0(dense_100/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d 2
dense_100/BiasAddz
dense_100/ReluReludense_100/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d 2
dense_100/Relu?
"dense_101/Tensordot/ReadVariableOpReadVariableOp+dense_101_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02$
"dense_101/Tensordot/ReadVariableOp~
dense_101/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_101/Tensordot/axes?
dense_101/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_101/Tensordot/free?
dense_101/Tensordot/ShapeShapedense_100/Relu:activations:0*
T0*
_output_shapes
:2
dense_101/Tensordot/Shape?
!dense_101/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_101/Tensordot/GatherV2/axis?
dense_101/Tensordot/GatherV2GatherV2"dense_101/Tensordot/Shape:output:0!dense_101/Tensordot/free:output:0*dense_101/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_101/Tensordot/GatherV2?
#dense_101/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#dense_101/Tensordot/GatherV2_1/axis?
dense_101/Tensordot/GatherV2_1GatherV2"dense_101/Tensordot/Shape:output:0!dense_101/Tensordot/axes:output:0,dense_101/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
dense_101/Tensordot/GatherV2_1?
dense_101/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_101/Tensordot/Const?
dense_101/Tensordot/ProdProd%dense_101/Tensordot/GatherV2:output:0"dense_101/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_101/Tensordot/Prod?
dense_101/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_101/Tensordot/Const_1?
dense_101/Tensordot/Prod_1Prod'dense_101/Tensordot/GatherV2_1:output:0$dense_101/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_101/Tensordot/Prod_1?
dense_101/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_101/Tensordot/concat/axis?
dense_101/Tensordot/concatConcatV2!dense_101/Tensordot/free:output:0!dense_101/Tensordot/axes:output:0(dense_101/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_101/Tensordot/concat?
dense_101/Tensordot/stackPack!dense_101/Tensordot/Prod:output:0#dense_101/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_101/Tensordot/stack?
dense_101/Tensordot/transpose	Transposedense_100/Relu:activations:0#dense_101/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????d 2
dense_101/Tensordot/transpose?
dense_101/Tensordot/ReshapeReshape!dense_101/Tensordot/transpose:y:0"dense_101/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_101/Tensordot/Reshape?
dense_101/Tensordot/MatMulMatMul$dense_101/Tensordot/Reshape:output:0*dense_101/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_101/Tensordot/MatMul?
dense_101/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_101/Tensordot/Const_2?
!dense_101/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_101/Tensordot/concat_1/axis?
dense_101/Tensordot/concat_1ConcatV2%dense_101/Tensordot/GatherV2:output:0$dense_101/Tensordot/Const_2:output:0*dense_101/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_101/Tensordot/concat_1?
dense_101/TensordotReshape$dense_101/Tensordot/MatMul:product:0%dense_101/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????d@2
dense_101/Tensordot?
 dense_101/BiasAdd/ReadVariableOpReadVariableOp)dense_101_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 dense_101/BiasAdd/ReadVariableOp?
dense_101/BiasAddBiasAdddense_101/Tensordot:output:0(dense_101/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@2
dense_101/BiasAdd?
IdentityIdentitydense_101/BiasAdd:output:0!^dense_100/BiasAdd/ReadVariableOp#^dense_100/Tensordot/ReadVariableOp!^dense_101/BiasAdd/ReadVariableOp#^dense_101/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????d@: : : : 2D
 dense_100/BiasAdd/ReadVariableOp dense_100/BiasAdd/ReadVariableOp2H
"dense_100/Tensordot/ReadVariableOp"dense_100/Tensordot/ReadVariableOp2D
 dense_101/BiasAdd/ReadVariableOp dense_101/BiasAdd/ReadVariableOp2H
"dense_101/Tensordot/ReadVariableOp"dense_101/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????d@
 
_user_specified_nameinputs
??
?
N__inference_transformer_block_8_layer_call_and_return_conditional_losses_93270

inputsS
Amulti_head_attention_8_dense_96_tensordot_readvariableop_resource:@@M
?multi_head_attention_8_dense_96_biasadd_readvariableop_resource:@S
Amulti_head_attention_8_dense_97_tensordot_readvariableop_resource:@@M
?multi_head_attention_8_dense_97_biasadd_readvariableop_resource:@S
Amulti_head_attention_8_dense_98_tensordot_readvariableop_resource:@@M
?multi_head_attention_8_dense_98_biasadd_readvariableop_resource:@S
Amulti_head_attention_8_dense_99_tensordot_readvariableop_resource:@@M
?multi_head_attention_8_dense_99_biasadd_readvariableop_resource:@J
<layer_normalization_26_batchnorm_mul_readvariableop_resource:@F
8layer_normalization_26_batchnorm_readvariableop_resource:@J
8sequential_8_dense_100_tensordot_readvariableop_resource:@ D
6sequential_8_dense_100_biasadd_readvariableop_resource: J
8sequential_8_dense_101_tensordot_readvariableop_resource: @D
6sequential_8_dense_101_biasadd_readvariableop_resource:@J
<layer_normalization_27_batchnorm_mul_readvariableop_resource:@F
8layer_normalization_27_batchnorm_readvariableop_resource:@
identity??/layer_normalization_26/batchnorm/ReadVariableOp?3layer_normalization_26/batchnorm/mul/ReadVariableOp?/layer_normalization_27/batchnorm/ReadVariableOp?3layer_normalization_27/batchnorm/mul/ReadVariableOp?6multi_head_attention_8/dense_96/BiasAdd/ReadVariableOp?8multi_head_attention_8/dense_96/Tensordot/ReadVariableOp?6multi_head_attention_8/dense_97/BiasAdd/ReadVariableOp?8multi_head_attention_8/dense_97/Tensordot/ReadVariableOp?6multi_head_attention_8/dense_98/BiasAdd/ReadVariableOp?8multi_head_attention_8/dense_98/Tensordot/ReadVariableOp?6multi_head_attention_8/dense_99/BiasAdd/ReadVariableOp?8multi_head_attention_8/dense_99/Tensordot/ReadVariableOp?-sequential_8/dense_100/BiasAdd/ReadVariableOp?/sequential_8/dense_100/Tensordot/ReadVariableOp?-sequential_8/dense_101/BiasAdd/ReadVariableOp?/sequential_8/dense_101/Tensordot/ReadVariableOpr
multi_head_attention_8/ShapeShapeinputs*
T0*
_output_shapes
:2
multi_head_attention_8/Shape?
*multi_head_attention_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*multi_head_attention_8/strided_slice/stack?
,multi_head_attention_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,multi_head_attention_8/strided_slice/stack_1?
,multi_head_attention_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,multi_head_attention_8/strided_slice/stack_2?
$multi_head_attention_8/strided_sliceStridedSlice%multi_head_attention_8/Shape:output:03multi_head_attention_8/strided_slice/stack:output:05multi_head_attention_8/strided_slice/stack_1:output:05multi_head_attention_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$multi_head_attention_8/strided_slice?
8multi_head_attention_8/dense_96/Tensordot/ReadVariableOpReadVariableOpAmulti_head_attention_8_dense_96_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02:
8multi_head_attention_8/dense_96/Tensordot/ReadVariableOp?
.multi_head_attention_8/dense_96/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:20
.multi_head_attention_8/dense_96/Tensordot/axes?
.multi_head_attention_8/dense_96/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       20
.multi_head_attention_8/dense_96/Tensordot/free?
/multi_head_attention_8/dense_96/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:21
/multi_head_attention_8/dense_96/Tensordot/Shape?
7multi_head_attention_8/dense_96/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_attention_8/dense_96/Tensordot/GatherV2/axis?
2multi_head_attention_8/dense_96/Tensordot/GatherV2GatherV28multi_head_attention_8/dense_96/Tensordot/Shape:output:07multi_head_attention_8/dense_96/Tensordot/free:output:0@multi_head_attention_8/dense_96/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:24
2multi_head_attention_8/dense_96/Tensordot/GatherV2?
9multi_head_attention_8/dense_96/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9multi_head_attention_8/dense_96/Tensordot/GatherV2_1/axis?
4multi_head_attention_8/dense_96/Tensordot/GatherV2_1GatherV28multi_head_attention_8/dense_96/Tensordot/Shape:output:07multi_head_attention_8/dense_96/Tensordot/axes:output:0Bmulti_head_attention_8/dense_96/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4multi_head_attention_8/dense_96/Tensordot/GatherV2_1?
/multi_head_attention_8/dense_96/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/multi_head_attention_8/dense_96/Tensordot/Const?
.multi_head_attention_8/dense_96/Tensordot/ProdProd;multi_head_attention_8/dense_96/Tensordot/GatherV2:output:08multi_head_attention_8/dense_96/Tensordot/Const:output:0*
T0*
_output_shapes
: 20
.multi_head_attention_8/dense_96/Tensordot/Prod?
1multi_head_attention_8/dense_96/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1multi_head_attention_8/dense_96/Tensordot/Const_1?
0multi_head_attention_8/dense_96/Tensordot/Prod_1Prod=multi_head_attention_8/dense_96/Tensordot/GatherV2_1:output:0:multi_head_attention_8/dense_96/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 22
0multi_head_attention_8/dense_96/Tensordot/Prod_1?
5multi_head_attention_8/dense_96/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5multi_head_attention_8/dense_96/Tensordot/concat/axis?
0multi_head_attention_8/dense_96/Tensordot/concatConcatV27multi_head_attention_8/dense_96/Tensordot/free:output:07multi_head_attention_8/dense_96/Tensordot/axes:output:0>multi_head_attention_8/dense_96/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:22
0multi_head_attention_8/dense_96/Tensordot/concat?
/multi_head_attention_8/dense_96/Tensordot/stackPack7multi_head_attention_8/dense_96/Tensordot/Prod:output:09multi_head_attention_8/dense_96/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:21
/multi_head_attention_8/dense_96/Tensordot/stack?
3multi_head_attention_8/dense_96/Tensordot/transpose	Transposeinputs9multi_head_attention_8/dense_96/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????d@25
3multi_head_attention_8/dense_96/Tensordot/transpose?
1multi_head_attention_8/dense_96/Tensordot/ReshapeReshape7multi_head_attention_8/dense_96/Tensordot/transpose:y:08multi_head_attention_8/dense_96/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????23
1multi_head_attention_8/dense_96/Tensordot/Reshape?
0multi_head_attention_8/dense_96/Tensordot/MatMulMatMul:multi_head_attention_8/dense_96/Tensordot/Reshape:output:0@multi_head_attention_8/dense_96/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@22
0multi_head_attention_8/dense_96/Tensordot/MatMul?
1multi_head_attention_8/dense_96/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@23
1multi_head_attention_8/dense_96/Tensordot/Const_2?
7multi_head_attention_8/dense_96/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_attention_8/dense_96/Tensordot/concat_1/axis?
2multi_head_attention_8/dense_96/Tensordot/concat_1ConcatV2;multi_head_attention_8/dense_96/Tensordot/GatherV2:output:0:multi_head_attention_8/dense_96/Tensordot/Const_2:output:0@multi_head_attention_8/dense_96/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:24
2multi_head_attention_8/dense_96/Tensordot/concat_1?
)multi_head_attention_8/dense_96/TensordotReshape:multi_head_attention_8/dense_96/Tensordot/MatMul:product:0;multi_head_attention_8/dense_96/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????d@2+
)multi_head_attention_8/dense_96/Tensordot?
6multi_head_attention_8/dense_96/BiasAdd/ReadVariableOpReadVariableOp?multi_head_attention_8_dense_96_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6multi_head_attention_8/dense_96/BiasAdd/ReadVariableOp?
'multi_head_attention_8/dense_96/BiasAddBiasAdd2multi_head_attention_8/dense_96/Tensordot:output:0>multi_head_attention_8/dense_96/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@2)
'multi_head_attention_8/dense_96/BiasAdd?
8multi_head_attention_8/dense_97/Tensordot/ReadVariableOpReadVariableOpAmulti_head_attention_8_dense_97_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02:
8multi_head_attention_8/dense_97/Tensordot/ReadVariableOp?
.multi_head_attention_8/dense_97/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:20
.multi_head_attention_8/dense_97/Tensordot/axes?
.multi_head_attention_8/dense_97/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       20
.multi_head_attention_8/dense_97/Tensordot/free?
/multi_head_attention_8/dense_97/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:21
/multi_head_attention_8/dense_97/Tensordot/Shape?
7multi_head_attention_8/dense_97/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_attention_8/dense_97/Tensordot/GatherV2/axis?
2multi_head_attention_8/dense_97/Tensordot/GatherV2GatherV28multi_head_attention_8/dense_97/Tensordot/Shape:output:07multi_head_attention_8/dense_97/Tensordot/free:output:0@multi_head_attention_8/dense_97/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:24
2multi_head_attention_8/dense_97/Tensordot/GatherV2?
9multi_head_attention_8/dense_97/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9multi_head_attention_8/dense_97/Tensordot/GatherV2_1/axis?
4multi_head_attention_8/dense_97/Tensordot/GatherV2_1GatherV28multi_head_attention_8/dense_97/Tensordot/Shape:output:07multi_head_attention_8/dense_97/Tensordot/axes:output:0Bmulti_head_attention_8/dense_97/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4multi_head_attention_8/dense_97/Tensordot/GatherV2_1?
/multi_head_attention_8/dense_97/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/multi_head_attention_8/dense_97/Tensordot/Const?
.multi_head_attention_8/dense_97/Tensordot/ProdProd;multi_head_attention_8/dense_97/Tensordot/GatherV2:output:08multi_head_attention_8/dense_97/Tensordot/Const:output:0*
T0*
_output_shapes
: 20
.multi_head_attention_8/dense_97/Tensordot/Prod?
1multi_head_attention_8/dense_97/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1multi_head_attention_8/dense_97/Tensordot/Const_1?
0multi_head_attention_8/dense_97/Tensordot/Prod_1Prod=multi_head_attention_8/dense_97/Tensordot/GatherV2_1:output:0:multi_head_attention_8/dense_97/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 22
0multi_head_attention_8/dense_97/Tensordot/Prod_1?
5multi_head_attention_8/dense_97/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5multi_head_attention_8/dense_97/Tensordot/concat/axis?
0multi_head_attention_8/dense_97/Tensordot/concatConcatV27multi_head_attention_8/dense_97/Tensordot/free:output:07multi_head_attention_8/dense_97/Tensordot/axes:output:0>multi_head_attention_8/dense_97/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:22
0multi_head_attention_8/dense_97/Tensordot/concat?
/multi_head_attention_8/dense_97/Tensordot/stackPack7multi_head_attention_8/dense_97/Tensordot/Prod:output:09multi_head_attention_8/dense_97/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:21
/multi_head_attention_8/dense_97/Tensordot/stack?
3multi_head_attention_8/dense_97/Tensordot/transpose	Transposeinputs9multi_head_attention_8/dense_97/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????d@25
3multi_head_attention_8/dense_97/Tensordot/transpose?
1multi_head_attention_8/dense_97/Tensordot/ReshapeReshape7multi_head_attention_8/dense_97/Tensordot/transpose:y:08multi_head_attention_8/dense_97/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????23
1multi_head_attention_8/dense_97/Tensordot/Reshape?
0multi_head_attention_8/dense_97/Tensordot/MatMulMatMul:multi_head_attention_8/dense_97/Tensordot/Reshape:output:0@multi_head_attention_8/dense_97/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@22
0multi_head_attention_8/dense_97/Tensordot/MatMul?
1multi_head_attention_8/dense_97/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@23
1multi_head_attention_8/dense_97/Tensordot/Const_2?
7multi_head_attention_8/dense_97/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_attention_8/dense_97/Tensordot/concat_1/axis?
2multi_head_attention_8/dense_97/Tensordot/concat_1ConcatV2;multi_head_attention_8/dense_97/Tensordot/GatherV2:output:0:multi_head_attention_8/dense_97/Tensordot/Const_2:output:0@multi_head_attention_8/dense_97/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:24
2multi_head_attention_8/dense_97/Tensordot/concat_1?
)multi_head_attention_8/dense_97/TensordotReshape:multi_head_attention_8/dense_97/Tensordot/MatMul:product:0;multi_head_attention_8/dense_97/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????d@2+
)multi_head_attention_8/dense_97/Tensordot?
6multi_head_attention_8/dense_97/BiasAdd/ReadVariableOpReadVariableOp?multi_head_attention_8_dense_97_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6multi_head_attention_8/dense_97/BiasAdd/ReadVariableOp?
'multi_head_attention_8/dense_97/BiasAddBiasAdd2multi_head_attention_8/dense_97/Tensordot:output:0>multi_head_attention_8/dense_97/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@2)
'multi_head_attention_8/dense_97/BiasAdd?
8multi_head_attention_8/dense_98/Tensordot/ReadVariableOpReadVariableOpAmulti_head_attention_8_dense_98_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02:
8multi_head_attention_8/dense_98/Tensordot/ReadVariableOp?
.multi_head_attention_8/dense_98/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:20
.multi_head_attention_8/dense_98/Tensordot/axes?
.multi_head_attention_8/dense_98/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       20
.multi_head_attention_8/dense_98/Tensordot/free?
/multi_head_attention_8/dense_98/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:21
/multi_head_attention_8/dense_98/Tensordot/Shape?
7multi_head_attention_8/dense_98/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_attention_8/dense_98/Tensordot/GatherV2/axis?
2multi_head_attention_8/dense_98/Tensordot/GatherV2GatherV28multi_head_attention_8/dense_98/Tensordot/Shape:output:07multi_head_attention_8/dense_98/Tensordot/free:output:0@multi_head_attention_8/dense_98/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:24
2multi_head_attention_8/dense_98/Tensordot/GatherV2?
9multi_head_attention_8/dense_98/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9multi_head_attention_8/dense_98/Tensordot/GatherV2_1/axis?
4multi_head_attention_8/dense_98/Tensordot/GatherV2_1GatherV28multi_head_attention_8/dense_98/Tensordot/Shape:output:07multi_head_attention_8/dense_98/Tensordot/axes:output:0Bmulti_head_attention_8/dense_98/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4multi_head_attention_8/dense_98/Tensordot/GatherV2_1?
/multi_head_attention_8/dense_98/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/multi_head_attention_8/dense_98/Tensordot/Const?
.multi_head_attention_8/dense_98/Tensordot/ProdProd;multi_head_attention_8/dense_98/Tensordot/GatherV2:output:08multi_head_attention_8/dense_98/Tensordot/Const:output:0*
T0*
_output_shapes
: 20
.multi_head_attention_8/dense_98/Tensordot/Prod?
1multi_head_attention_8/dense_98/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1multi_head_attention_8/dense_98/Tensordot/Const_1?
0multi_head_attention_8/dense_98/Tensordot/Prod_1Prod=multi_head_attention_8/dense_98/Tensordot/GatherV2_1:output:0:multi_head_attention_8/dense_98/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 22
0multi_head_attention_8/dense_98/Tensordot/Prod_1?
5multi_head_attention_8/dense_98/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5multi_head_attention_8/dense_98/Tensordot/concat/axis?
0multi_head_attention_8/dense_98/Tensordot/concatConcatV27multi_head_attention_8/dense_98/Tensordot/free:output:07multi_head_attention_8/dense_98/Tensordot/axes:output:0>multi_head_attention_8/dense_98/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:22
0multi_head_attention_8/dense_98/Tensordot/concat?
/multi_head_attention_8/dense_98/Tensordot/stackPack7multi_head_attention_8/dense_98/Tensordot/Prod:output:09multi_head_attention_8/dense_98/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:21
/multi_head_attention_8/dense_98/Tensordot/stack?
3multi_head_attention_8/dense_98/Tensordot/transpose	Transposeinputs9multi_head_attention_8/dense_98/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????d@25
3multi_head_attention_8/dense_98/Tensordot/transpose?
1multi_head_attention_8/dense_98/Tensordot/ReshapeReshape7multi_head_attention_8/dense_98/Tensordot/transpose:y:08multi_head_attention_8/dense_98/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????23
1multi_head_attention_8/dense_98/Tensordot/Reshape?
0multi_head_attention_8/dense_98/Tensordot/MatMulMatMul:multi_head_attention_8/dense_98/Tensordot/Reshape:output:0@multi_head_attention_8/dense_98/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@22
0multi_head_attention_8/dense_98/Tensordot/MatMul?
1multi_head_attention_8/dense_98/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@23
1multi_head_attention_8/dense_98/Tensordot/Const_2?
7multi_head_attention_8/dense_98/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_attention_8/dense_98/Tensordot/concat_1/axis?
2multi_head_attention_8/dense_98/Tensordot/concat_1ConcatV2;multi_head_attention_8/dense_98/Tensordot/GatherV2:output:0:multi_head_attention_8/dense_98/Tensordot/Const_2:output:0@multi_head_attention_8/dense_98/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:24
2multi_head_attention_8/dense_98/Tensordot/concat_1?
)multi_head_attention_8/dense_98/TensordotReshape:multi_head_attention_8/dense_98/Tensordot/MatMul:product:0;multi_head_attention_8/dense_98/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????d@2+
)multi_head_attention_8/dense_98/Tensordot?
6multi_head_attention_8/dense_98/BiasAdd/ReadVariableOpReadVariableOp?multi_head_attention_8_dense_98_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6multi_head_attention_8/dense_98/BiasAdd/ReadVariableOp?
'multi_head_attention_8/dense_98/BiasAddBiasAdd2multi_head_attention_8/dense_98/Tensordot:output:0>multi_head_attention_8/dense_98/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@2)
'multi_head_attention_8/dense_98/BiasAdd?
&multi_head_attention_8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&multi_head_attention_8/Reshape/shape/1?
&multi_head_attention_8/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2(
&multi_head_attention_8/Reshape/shape/2?
&multi_head_attention_8/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2(
&multi_head_attention_8/Reshape/shape/3?
$multi_head_attention_8/Reshape/shapePack-multi_head_attention_8/strided_slice:output:0/multi_head_attention_8/Reshape/shape/1:output:0/multi_head_attention_8/Reshape/shape/2:output:0/multi_head_attention_8/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2&
$multi_head_attention_8/Reshape/shape?
multi_head_attention_8/ReshapeReshape0multi_head_attention_8/dense_96/BiasAdd:output:0-multi_head_attention_8/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2 
multi_head_attention_8/Reshape?
%multi_head_attention_8/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2'
%multi_head_attention_8/transpose/perm?
 multi_head_attention_8/transpose	Transpose'multi_head_attention_8/Reshape:output:0.multi_head_attention_8/transpose/perm:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2"
 multi_head_attention_8/transpose?
(multi_head_attention_8/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(multi_head_attention_8/Reshape_1/shape/1?
(multi_head_attention_8/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(multi_head_attention_8/Reshape_1/shape/2?
(multi_head_attention_8/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2*
(multi_head_attention_8/Reshape_1/shape/3?
&multi_head_attention_8/Reshape_1/shapePack-multi_head_attention_8/strided_slice:output:01multi_head_attention_8/Reshape_1/shape/1:output:01multi_head_attention_8/Reshape_1/shape/2:output:01multi_head_attention_8/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_attention_8/Reshape_1/shape?
 multi_head_attention_8/Reshape_1Reshape0multi_head_attention_8/dense_97/BiasAdd:output:0/multi_head_attention_8/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2"
 multi_head_attention_8/Reshape_1?
'multi_head_attention_8/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2)
'multi_head_attention_8/transpose_1/perm?
"multi_head_attention_8/transpose_1	Transpose)multi_head_attention_8/Reshape_1:output:00multi_head_attention_8/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2$
"multi_head_attention_8/transpose_1?
(multi_head_attention_8/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(multi_head_attention_8/Reshape_2/shape/1?
(multi_head_attention_8/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(multi_head_attention_8/Reshape_2/shape/2?
(multi_head_attention_8/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2*
(multi_head_attention_8/Reshape_2/shape/3?
&multi_head_attention_8/Reshape_2/shapePack-multi_head_attention_8/strided_slice:output:01multi_head_attention_8/Reshape_2/shape/1:output:01multi_head_attention_8/Reshape_2/shape/2:output:01multi_head_attention_8/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_attention_8/Reshape_2/shape?
 multi_head_attention_8/Reshape_2Reshape0multi_head_attention_8/dense_98/BiasAdd:output:0/multi_head_attention_8/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2"
 multi_head_attention_8/Reshape_2?
'multi_head_attention_8/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2)
'multi_head_attention_8/transpose_2/perm?
"multi_head_attention_8/transpose_2	Transpose)multi_head_attention_8/Reshape_2:output:00multi_head_attention_8/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2$
"multi_head_attention_8/transpose_2?
multi_head_attention_8/MatMulBatchMatMulV2$multi_head_attention_8/transpose:y:0&multi_head_attention_8/transpose_1:y:0*
T0*A
_output_shapes/
-:+???????????????????????????*
adj_y(2
multi_head_attention_8/MatMul?
multi_head_attention_8/Shape_1Shape&multi_head_attention_8/transpose_1:y:0*
T0*
_output_shapes
:2 
multi_head_attention_8/Shape_1?
,multi_head_attention_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2.
,multi_head_attention_8/strided_slice_1/stack?
.multi_head_attention_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 20
.multi_head_attention_8/strided_slice_1/stack_1?
.multi_head_attention_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.multi_head_attention_8/strided_slice_1/stack_2?
&multi_head_attention_8/strided_slice_1StridedSlice'multi_head_attention_8/Shape_1:output:05multi_head_attention_8/strided_slice_1/stack:output:07multi_head_attention_8/strided_slice_1/stack_1:output:07multi_head_attention_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&multi_head_attention_8/strided_slice_1?
multi_head_attention_8/CastCast/multi_head_attention_8/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
multi_head_attention_8/Cast?
multi_head_attention_8/SqrtSqrtmulti_head_attention_8/Cast:y:0*
T0*
_output_shapes
: 2
multi_head_attention_8/Sqrt?
multi_head_attention_8/truedivRealDiv&multi_head_attention_8/MatMul:output:0multi_head_attention_8/Sqrt:y:0*
T0*A
_output_shapes/
-:+???????????????????????????2 
multi_head_attention_8/truediv?
multi_head_attention_8/SoftmaxSoftmax"multi_head_attention_8/truediv:z:0*
T0*A
_output_shapes/
-:+???????????????????????????2 
multi_head_attention_8/Softmax?
multi_head_attention_8/MatMul_1BatchMatMulV2(multi_head_attention_8/Softmax:softmax:0&multi_head_attention_8/transpose_2:y:0*
T0*8
_output_shapes&
$:"?????????????????? 2!
multi_head_attention_8/MatMul_1?
'multi_head_attention_8/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             2)
'multi_head_attention_8/transpose_3/perm?
"multi_head_attention_8/transpose_3	Transpose(multi_head_attention_8/MatMul_1:output:00multi_head_attention_8/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2$
"multi_head_attention_8/transpose_3?
(multi_head_attention_8/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(multi_head_attention_8/Reshape_3/shape/1?
(multi_head_attention_8/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@2*
(multi_head_attention_8/Reshape_3/shape/2?
&multi_head_attention_8/Reshape_3/shapePack-multi_head_attention_8/strided_slice:output:01multi_head_attention_8/Reshape_3/shape/1:output:01multi_head_attention_8/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_attention_8/Reshape_3/shape?
 multi_head_attention_8/Reshape_3Reshape&multi_head_attention_8/transpose_3:y:0/multi_head_attention_8/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :??????????????????@2"
 multi_head_attention_8/Reshape_3?
8multi_head_attention_8/dense_99/Tensordot/ReadVariableOpReadVariableOpAmulti_head_attention_8_dense_99_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02:
8multi_head_attention_8/dense_99/Tensordot/ReadVariableOp?
.multi_head_attention_8/dense_99/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:20
.multi_head_attention_8/dense_99/Tensordot/axes?
.multi_head_attention_8/dense_99/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       20
.multi_head_attention_8/dense_99/Tensordot/free?
/multi_head_attention_8/dense_99/Tensordot/ShapeShape)multi_head_attention_8/Reshape_3:output:0*
T0*
_output_shapes
:21
/multi_head_attention_8/dense_99/Tensordot/Shape?
7multi_head_attention_8/dense_99/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_attention_8/dense_99/Tensordot/GatherV2/axis?
2multi_head_attention_8/dense_99/Tensordot/GatherV2GatherV28multi_head_attention_8/dense_99/Tensordot/Shape:output:07multi_head_attention_8/dense_99/Tensordot/free:output:0@multi_head_attention_8/dense_99/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:24
2multi_head_attention_8/dense_99/Tensordot/GatherV2?
9multi_head_attention_8/dense_99/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9multi_head_attention_8/dense_99/Tensordot/GatherV2_1/axis?
4multi_head_attention_8/dense_99/Tensordot/GatherV2_1GatherV28multi_head_attention_8/dense_99/Tensordot/Shape:output:07multi_head_attention_8/dense_99/Tensordot/axes:output:0Bmulti_head_attention_8/dense_99/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4multi_head_attention_8/dense_99/Tensordot/GatherV2_1?
/multi_head_attention_8/dense_99/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/multi_head_attention_8/dense_99/Tensordot/Const?
.multi_head_attention_8/dense_99/Tensordot/ProdProd;multi_head_attention_8/dense_99/Tensordot/GatherV2:output:08multi_head_attention_8/dense_99/Tensordot/Const:output:0*
T0*
_output_shapes
: 20
.multi_head_attention_8/dense_99/Tensordot/Prod?
1multi_head_attention_8/dense_99/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1multi_head_attention_8/dense_99/Tensordot/Const_1?
0multi_head_attention_8/dense_99/Tensordot/Prod_1Prod=multi_head_attention_8/dense_99/Tensordot/GatherV2_1:output:0:multi_head_attention_8/dense_99/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 22
0multi_head_attention_8/dense_99/Tensordot/Prod_1?
5multi_head_attention_8/dense_99/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5multi_head_attention_8/dense_99/Tensordot/concat/axis?
0multi_head_attention_8/dense_99/Tensordot/concatConcatV27multi_head_attention_8/dense_99/Tensordot/free:output:07multi_head_attention_8/dense_99/Tensordot/axes:output:0>multi_head_attention_8/dense_99/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:22
0multi_head_attention_8/dense_99/Tensordot/concat?
/multi_head_attention_8/dense_99/Tensordot/stackPack7multi_head_attention_8/dense_99/Tensordot/Prod:output:09multi_head_attention_8/dense_99/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:21
/multi_head_attention_8/dense_99/Tensordot/stack?
3multi_head_attention_8/dense_99/Tensordot/transpose	Transpose)multi_head_attention_8/Reshape_3:output:09multi_head_attention_8/dense_99/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :??????????????????@25
3multi_head_attention_8/dense_99/Tensordot/transpose?
1multi_head_attention_8/dense_99/Tensordot/ReshapeReshape7multi_head_attention_8/dense_99/Tensordot/transpose:y:08multi_head_attention_8/dense_99/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????23
1multi_head_attention_8/dense_99/Tensordot/Reshape?
0multi_head_attention_8/dense_99/Tensordot/MatMulMatMul:multi_head_attention_8/dense_99/Tensordot/Reshape:output:0@multi_head_attention_8/dense_99/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@22
0multi_head_attention_8/dense_99/Tensordot/MatMul?
1multi_head_attention_8/dense_99/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@23
1multi_head_attention_8/dense_99/Tensordot/Const_2?
7multi_head_attention_8/dense_99/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_attention_8/dense_99/Tensordot/concat_1/axis?
2multi_head_attention_8/dense_99/Tensordot/concat_1ConcatV2;multi_head_attention_8/dense_99/Tensordot/GatherV2:output:0:multi_head_attention_8/dense_99/Tensordot/Const_2:output:0@multi_head_attention_8/dense_99/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:24
2multi_head_attention_8/dense_99/Tensordot/concat_1?
)multi_head_attention_8/dense_99/TensordotReshape:multi_head_attention_8/dense_99/Tensordot/MatMul:product:0;multi_head_attention_8/dense_99/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :??????????????????@2+
)multi_head_attention_8/dense_99/Tensordot?
6multi_head_attention_8/dense_99/BiasAdd/ReadVariableOpReadVariableOp?multi_head_attention_8_dense_99_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6multi_head_attention_8/dense_99/BiasAdd/ReadVariableOp?
'multi_head_attention_8/dense_99/BiasAddBiasAdd2multi_head_attention_8/dense_99/Tensordot:output:0>multi_head_attention_8/dense_99/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????@2)
'multi_head_attention_8/dense_99/BiasAdd?
dropout_42/IdentityIdentity0multi_head_attention_8/dense_99/BiasAdd:output:0*
T0*4
_output_shapes"
 :??????????????????@2
dropout_42/Identityo
addAddV2inputsdropout_42/Identity:output:0*
T0*+
_output_shapes
:?????????d@2
add?
5layer_normalization_26/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_26/moments/mean/reduction_indices?
#layer_normalization_26/moments/meanMeanadd:z:0>layer_normalization_26/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????d*
	keep_dims(2%
#layer_normalization_26/moments/mean?
+layer_normalization_26/moments/StopGradientStopGradient,layer_normalization_26/moments/mean:output:0*
T0*+
_output_shapes
:?????????d2-
+layer_normalization_26/moments/StopGradient?
0layer_normalization_26/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_26/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????d@22
0layer_normalization_26/moments/SquaredDifference?
9layer_normalization_26/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_26/moments/variance/reduction_indices?
'layer_normalization_26/moments/varianceMean4layer_normalization_26/moments/SquaredDifference:z:0Blayer_normalization_26/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????d*
	keep_dims(2)
'layer_normalization_26/moments/variance?
&layer_normalization_26/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52(
&layer_normalization_26/batchnorm/add/y?
$layer_normalization_26/batchnorm/addAddV20layer_normalization_26/moments/variance:output:0/layer_normalization_26/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????d2&
$layer_normalization_26/batchnorm/add?
&layer_normalization_26/batchnorm/RsqrtRsqrt(layer_normalization_26/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????d2(
&layer_normalization_26/batchnorm/Rsqrt?
3layer_normalization_26/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_26_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype025
3layer_normalization_26/batchnorm/mul/ReadVariableOp?
$layer_normalization_26/batchnorm/mulMul*layer_normalization_26/batchnorm/Rsqrt:y:0;layer_normalization_26/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@2&
$layer_normalization_26/batchnorm/mul?
&layer_normalization_26/batchnorm/mul_1Muladd:z:0(layer_normalization_26/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????d@2(
&layer_normalization_26/batchnorm/mul_1?
&layer_normalization_26/batchnorm/mul_2Mul,layer_normalization_26/moments/mean:output:0(layer_normalization_26/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????d@2(
&layer_normalization_26/batchnorm/mul_2?
/layer_normalization_26/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_26_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype021
/layer_normalization_26/batchnorm/ReadVariableOp?
$layer_normalization_26/batchnorm/subSub7layer_normalization_26/batchnorm/ReadVariableOp:value:0*layer_normalization_26/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????d@2&
$layer_normalization_26/batchnorm/sub?
&layer_normalization_26/batchnorm/add_1AddV2*layer_normalization_26/batchnorm/mul_1:z:0(layer_normalization_26/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????d@2(
&layer_normalization_26/batchnorm/add_1?
/sequential_8/dense_100/Tensordot/ReadVariableOpReadVariableOp8sequential_8_dense_100_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype021
/sequential_8/dense_100/Tensordot/ReadVariableOp?
%sequential_8/dense_100/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_8/dense_100/Tensordot/axes?
%sequential_8/dense_100/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_8/dense_100/Tensordot/free?
&sequential_8/dense_100/Tensordot/ShapeShape*layer_normalization_26/batchnorm/add_1:z:0*
T0*
_output_shapes
:2(
&sequential_8/dense_100/Tensordot/Shape?
.sequential_8/dense_100/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_8/dense_100/Tensordot/GatherV2/axis?
)sequential_8/dense_100/Tensordot/GatherV2GatherV2/sequential_8/dense_100/Tensordot/Shape:output:0.sequential_8/dense_100/Tensordot/free:output:07sequential_8/dense_100/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_8/dense_100/Tensordot/GatherV2?
0sequential_8/dense_100/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_8/dense_100/Tensordot/GatherV2_1/axis?
+sequential_8/dense_100/Tensordot/GatherV2_1GatherV2/sequential_8/dense_100/Tensordot/Shape:output:0.sequential_8/dense_100/Tensordot/axes:output:09sequential_8/dense_100/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_8/dense_100/Tensordot/GatherV2_1?
&sequential_8/dense_100/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_8/dense_100/Tensordot/Const?
%sequential_8/dense_100/Tensordot/ProdProd2sequential_8/dense_100/Tensordot/GatherV2:output:0/sequential_8/dense_100/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_8/dense_100/Tensordot/Prod?
(sequential_8/dense_100/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_8/dense_100/Tensordot/Const_1?
'sequential_8/dense_100/Tensordot/Prod_1Prod4sequential_8/dense_100/Tensordot/GatherV2_1:output:01sequential_8/dense_100/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_8/dense_100/Tensordot/Prod_1?
,sequential_8/dense_100/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_8/dense_100/Tensordot/concat/axis?
'sequential_8/dense_100/Tensordot/concatConcatV2.sequential_8/dense_100/Tensordot/free:output:0.sequential_8/dense_100/Tensordot/axes:output:05sequential_8/dense_100/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_8/dense_100/Tensordot/concat?
&sequential_8/dense_100/Tensordot/stackPack.sequential_8/dense_100/Tensordot/Prod:output:00sequential_8/dense_100/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_8/dense_100/Tensordot/stack?
*sequential_8/dense_100/Tensordot/transpose	Transpose*layer_normalization_26/batchnorm/add_1:z:00sequential_8/dense_100/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????d@2,
*sequential_8/dense_100/Tensordot/transpose?
(sequential_8/dense_100/Tensordot/ReshapeReshape.sequential_8/dense_100/Tensordot/transpose:y:0/sequential_8/dense_100/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(sequential_8/dense_100/Tensordot/Reshape?
'sequential_8/dense_100/Tensordot/MatMulMatMul1sequential_8/dense_100/Tensordot/Reshape:output:07sequential_8/dense_100/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2)
'sequential_8/dense_100/Tensordot/MatMul?
(sequential_8/dense_100/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_8/dense_100/Tensordot/Const_2?
.sequential_8/dense_100/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_8/dense_100/Tensordot/concat_1/axis?
)sequential_8/dense_100/Tensordot/concat_1ConcatV22sequential_8/dense_100/Tensordot/GatherV2:output:01sequential_8/dense_100/Tensordot/Const_2:output:07sequential_8/dense_100/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_8/dense_100/Tensordot/concat_1?
 sequential_8/dense_100/TensordotReshape1sequential_8/dense_100/Tensordot/MatMul:product:02sequential_8/dense_100/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????d 2"
 sequential_8/dense_100/Tensordot?
-sequential_8/dense_100/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_dense_100_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_8/dense_100/BiasAdd/ReadVariableOp?
sequential_8/dense_100/BiasAddBiasAdd)sequential_8/dense_100/Tensordot:output:05sequential_8/dense_100/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d 2 
sequential_8/dense_100/BiasAdd?
sequential_8/dense_100/ReluRelu'sequential_8/dense_100/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d 2
sequential_8/dense_100/Relu?
/sequential_8/dense_101/Tensordot/ReadVariableOpReadVariableOp8sequential_8_dense_101_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype021
/sequential_8/dense_101/Tensordot/ReadVariableOp?
%sequential_8/dense_101/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_8/dense_101/Tensordot/axes?
%sequential_8/dense_101/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_8/dense_101/Tensordot/free?
&sequential_8/dense_101/Tensordot/ShapeShape)sequential_8/dense_100/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_8/dense_101/Tensordot/Shape?
.sequential_8/dense_101/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_8/dense_101/Tensordot/GatherV2/axis?
)sequential_8/dense_101/Tensordot/GatherV2GatherV2/sequential_8/dense_101/Tensordot/Shape:output:0.sequential_8/dense_101/Tensordot/free:output:07sequential_8/dense_101/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_8/dense_101/Tensordot/GatherV2?
0sequential_8/dense_101/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_8/dense_101/Tensordot/GatherV2_1/axis?
+sequential_8/dense_101/Tensordot/GatherV2_1GatherV2/sequential_8/dense_101/Tensordot/Shape:output:0.sequential_8/dense_101/Tensordot/axes:output:09sequential_8/dense_101/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_8/dense_101/Tensordot/GatherV2_1?
&sequential_8/dense_101/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_8/dense_101/Tensordot/Const?
%sequential_8/dense_101/Tensordot/ProdProd2sequential_8/dense_101/Tensordot/GatherV2:output:0/sequential_8/dense_101/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_8/dense_101/Tensordot/Prod?
(sequential_8/dense_101/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_8/dense_101/Tensordot/Const_1?
'sequential_8/dense_101/Tensordot/Prod_1Prod4sequential_8/dense_101/Tensordot/GatherV2_1:output:01sequential_8/dense_101/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_8/dense_101/Tensordot/Prod_1?
,sequential_8/dense_101/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_8/dense_101/Tensordot/concat/axis?
'sequential_8/dense_101/Tensordot/concatConcatV2.sequential_8/dense_101/Tensordot/free:output:0.sequential_8/dense_101/Tensordot/axes:output:05sequential_8/dense_101/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_8/dense_101/Tensordot/concat?
&sequential_8/dense_101/Tensordot/stackPack.sequential_8/dense_101/Tensordot/Prod:output:00sequential_8/dense_101/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_8/dense_101/Tensordot/stack?
*sequential_8/dense_101/Tensordot/transpose	Transpose)sequential_8/dense_100/Relu:activations:00sequential_8/dense_101/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????d 2,
*sequential_8/dense_101/Tensordot/transpose?
(sequential_8/dense_101/Tensordot/ReshapeReshape.sequential_8/dense_101/Tensordot/transpose:y:0/sequential_8/dense_101/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(sequential_8/dense_101/Tensordot/Reshape?
'sequential_8/dense_101/Tensordot/MatMulMatMul1sequential_8/dense_101/Tensordot/Reshape:output:07sequential_8/dense_101/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2)
'sequential_8/dense_101/Tensordot/MatMul?
(sequential_8/dense_101/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2*
(sequential_8/dense_101/Tensordot/Const_2?
.sequential_8/dense_101/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_8/dense_101/Tensordot/concat_1/axis?
)sequential_8/dense_101/Tensordot/concat_1ConcatV22sequential_8/dense_101/Tensordot/GatherV2:output:01sequential_8/dense_101/Tensordot/Const_2:output:07sequential_8/dense_101/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_8/dense_101/Tensordot/concat_1?
 sequential_8/dense_101/TensordotReshape1sequential_8/dense_101/Tensordot/MatMul:product:02sequential_8/dense_101/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????d@2"
 sequential_8/dense_101/Tensordot?
-sequential_8/dense_101/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_dense_101_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-sequential_8/dense_101/BiasAdd/ReadVariableOp?
sequential_8/dense_101/BiasAddBiasAdd)sequential_8/dense_101/Tensordot:output:05sequential_8/dense_101/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@2 
sequential_8/dense_101/BiasAdd?
dropout_43/IdentityIdentity'sequential_8/dense_101/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d@2
dropout_43/Identity?
add_1AddV2*layer_normalization_26/batchnorm/add_1:z:0dropout_43/Identity:output:0*
T0*+
_output_shapes
:?????????d@2
add_1?
5layer_normalization_27/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_27/moments/mean/reduction_indices?
#layer_normalization_27/moments/meanMean	add_1:z:0>layer_normalization_27/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????d*
	keep_dims(2%
#layer_normalization_27/moments/mean?
+layer_normalization_27/moments/StopGradientStopGradient,layer_normalization_27/moments/mean:output:0*
T0*+
_output_shapes
:?????????d2-
+layer_normalization_27/moments/StopGradient?
0layer_normalization_27/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_27/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????d@22
0layer_normalization_27/moments/SquaredDifference?
9layer_normalization_27/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_27/moments/variance/reduction_indices?
'layer_normalization_27/moments/varianceMean4layer_normalization_27/moments/SquaredDifference:z:0Blayer_normalization_27/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????d*
	keep_dims(2)
'layer_normalization_27/moments/variance?
&layer_normalization_27/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52(
&layer_normalization_27/batchnorm/add/y?
$layer_normalization_27/batchnorm/addAddV20layer_normalization_27/moments/variance:output:0/layer_normalization_27/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????d2&
$layer_normalization_27/batchnorm/add?
&layer_normalization_27/batchnorm/RsqrtRsqrt(layer_normalization_27/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????d2(
&layer_normalization_27/batchnorm/Rsqrt?
3layer_normalization_27/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_27_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype025
3layer_normalization_27/batchnorm/mul/ReadVariableOp?
$layer_normalization_27/batchnorm/mulMul*layer_normalization_27/batchnorm/Rsqrt:y:0;layer_normalization_27/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@2&
$layer_normalization_27/batchnorm/mul?
&layer_normalization_27/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_27/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????d@2(
&layer_normalization_27/batchnorm/mul_1?
&layer_normalization_27/batchnorm/mul_2Mul,layer_normalization_27/moments/mean:output:0(layer_normalization_27/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????d@2(
&layer_normalization_27/batchnorm/mul_2?
/layer_normalization_27/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_27_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype021
/layer_normalization_27/batchnorm/ReadVariableOp?
$layer_normalization_27/batchnorm/subSub7layer_normalization_27/batchnorm/ReadVariableOp:value:0*layer_normalization_27/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????d@2&
$layer_normalization_27/batchnorm/sub?
&layer_normalization_27/batchnorm/add_1AddV2*layer_normalization_27/batchnorm/mul_1:z:0(layer_normalization_27/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????d@2(
&layer_normalization_27/batchnorm/add_1?
IdentityIdentity*layer_normalization_27/batchnorm/add_1:z:00^layer_normalization_26/batchnorm/ReadVariableOp4^layer_normalization_26/batchnorm/mul/ReadVariableOp0^layer_normalization_27/batchnorm/ReadVariableOp4^layer_normalization_27/batchnorm/mul/ReadVariableOp7^multi_head_attention_8/dense_96/BiasAdd/ReadVariableOp9^multi_head_attention_8/dense_96/Tensordot/ReadVariableOp7^multi_head_attention_8/dense_97/BiasAdd/ReadVariableOp9^multi_head_attention_8/dense_97/Tensordot/ReadVariableOp7^multi_head_attention_8/dense_98/BiasAdd/ReadVariableOp9^multi_head_attention_8/dense_98/Tensordot/ReadVariableOp7^multi_head_attention_8/dense_99/BiasAdd/ReadVariableOp9^multi_head_attention_8/dense_99/Tensordot/ReadVariableOp.^sequential_8/dense_100/BiasAdd/ReadVariableOp0^sequential_8/dense_100/Tensordot/ReadVariableOp.^sequential_8/dense_101/BiasAdd/ReadVariableOp0^sequential_8/dense_101/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????d@: : : : : : : : : : : : : : : : 2b
/layer_normalization_26/batchnorm/ReadVariableOp/layer_normalization_26/batchnorm/ReadVariableOp2j
3layer_normalization_26/batchnorm/mul/ReadVariableOp3layer_normalization_26/batchnorm/mul/ReadVariableOp2b
/layer_normalization_27/batchnorm/ReadVariableOp/layer_normalization_27/batchnorm/ReadVariableOp2j
3layer_normalization_27/batchnorm/mul/ReadVariableOp3layer_normalization_27/batchnorm/mul/ReadVariableOp2p
6multi_head_attention_8/dense_96/BiasAdd/ReadVariableOp6multi_head_attention_8/dense_96/BiasAdd/ReadVariableOp2t
8multi_head_attention_8/dense_96/Tensordot/ReadVariableOp8multi_head_attention_8/dense_96/Tensordot/ReadVariableOp2p
6multi_head_attention_8/dense_97/BiasAdd/ReadVariableOp6multi_head_attention_8/dense_97/BiasAdd/ReadVariableOp2t
8multi_head_attention_8/dense_97/Tensordot/ReadVariableOp8multi_head_attention_8/dense_97/Tensordot/ReadVariableOp2p
6multi_head_attention_8/dense_98/BiasAdd/ReadVariableOp6multi_head_attention_8/dense_98/BiasAdd/ReadVariableOp2t
8multi_head_attention_8/dense_98/Tensordot/ReadVariableOp8multi_head_attention_8/dense_98/Tensordot/ReadVariableOp2p
6multi_head_attention_8/dense_99/BiasAdd/ReadVariableOp6multi_head_attention_8/dense_99/BiasAdd/ReadVariableOp2t
8multi_head_attention_8/dense_99/Tensordot/ReadVariableOp8multi_head_attention_8/dense_99/Tensordot/ReadVariableOp2^
-sequential_8/dense_100/BiasAdd/ReadVariableOp-sequential_8/dense_100/BiasAdd/ReadVariableOp2b
/sequential_8/dense_100/Tensordot/ReadVariableOp/sequential_8/dense_100/Tensordot/ReadVariableOp2^
-sequential_8/dense_101/BiasAdd/ReadVariableOp-sequential_8/dense_101/BiasAdd/ReadVariableOp2b
/sequential_8/dense_101/Tensordot/ReadVariableOp/sequential_8/dense_101/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????d@
 
_user_specified_nameinputs
??
?:
!__inference__traced_restore_94346
file_prefix3
!assignvariableop_dense_102_kernel:@/
!assignvariableop_1_dense_102_bias:5
#assignvariableop_2_dense_103_kernel:/
!assignvariableop_3_dense_103_bias:&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: /
%assignvariableop_8_adam_learning_rate: \
Iassignvariableop_9_token_and_position_embedding_9_embedding_20_embeddings:	?@@\
Jassignvariableop_10_token_and_position_embedding_9_embedding_21_embeddings:d@`
Nassignvariableop_11_transformer_block_8_multi_head_attention_8_dense_96_kernel:@@Z
Lassignvariableop_12_transformer_block_8_multi_head_attention_8_dense_96_bias:@`
Nassignvariableop_13_transformer_block_8_multi_head_attention_8_dense_97_kernel:@@Z
Lassignvariableop_14_transformer_block_8_multi_head_attention_8_dense_97_bias:@`
Nassignvariableop_15_transformer_block_8_multi_head_attention_8_dense_98_kernel:@@Z
Lassignvariableop_16_transformer_block_8_multi_head_attention_8_dense_98_bias:@`
Nassignvariableop_17_transformer_block_8_multi_head_attention_8_dense_99_kernel:@@Z
Lassignvariableop_18_transformer_block_8_multi_head_attention_8_dense_99_bias:@6
$assignvariableop_19_dense_100_kernel:@ 0
"assignvariableop_20_dense_100_bias: 6
$assignvariableop_21_dense_101_kernel: @0
"assignvariableop_22_dense_101_bias:@R
Dassignvariableop_23_transformer_block_8_layer_normalization_26_gamma:@Q
Cassignvariableop_24_transformer_block_8_layer_normalization_26_beta:@R
Dassignvariableop_25_transformer_block_8_layer_normalization_27_gamma:@Q
Cassignvariableop_26_transformer_block_8_layer_normalization_27_beta:@#
assignvariableop_27_total: #
assignvariableop_28_count: %
assignvariableop_29_total_1: %
assignvariableop_30_count_1: =
+assignvariableop_31_adam_dense_102_kernel_m:@7
)assignvariableop_32_adam_dense_102_bias_m:=
+assignvariableop_33_adam_dense_103_kernel_m:7
)assignvariableop_34_adam_dense_103_bias_m:d
Qassignvariableop_35_adam_token_and_position_embedding_9_embedding_20_embeddings_m:	?@@c
Qassignvariableop_36_adam_token_and_position_embedding_9_embedding_21_embeddings_m:d@g
Uassignvariableop_37_adam_transformer_block_8_multi_head_attention_8_dense_96_kernel_m:@@a
Sassignvariableop_38_adam_transformer_block_8_multi_head_attention_8_dense_96_bias_m:@g
Uassignvariableop_39_adam_transformer_block_8_multi_head_attention_8_dense_97_kernel_m:@@a
Sassignvariableop_40_adam_transformer_block_8_multi_head_attention_8_dense_97_bias_m:@g
Uassignvariableop_41_adam_transformer_block_8_multi_head_attention_8_dense_98_kernel_m:@@a
Sassignvariableop_42_adam_transformer_block_8_multi_head_attention_8_dense_98_bias_m:@g
Uassignvariableop_43_adam_transformer_block_8_multi_head_attention_8_dense_99_kernel_m:@@a
Sassignvariableop_44_adam_transformer_block_8_multi_head_attention_8_dense_99_bias_m:@=
+assignvariableop_45_adam_dense_100_kernel_m:@ 7
)assignvariableop_46_adam_dense_100_bias_m: =
+assignvariableop_47_adam_dense_101_kernel_m: @7
)assignvariableop_48_adam_dense_101_bias_m:@Y
Kassignvariableop_49_adam_transformer_block_8_layer_normalization_26_gamma_m:@X
Jassignvariableop_50_adam_transformer_block_8_layer_normalization_26_beta_m:@Y
Kassignvariableop_51_adam_transformer_block_8_layer_normalization_27_gamma_m:@X
Jassignvariableop_52_adam_transformer_block_8_layer_normalization_27_beta_m:@=
+assignvariableop_53_adam_dense_102_kernel_v:@7
)assignvariableop_54_adam_dense_102_bias_v:=
+assignvariableop_55_adam_dense_103_kernel_v:7
)assignvariableop_56_adam_dense_103_bias_v:d
Qassignvariableop_57_adam_token_and_position_embedding_9_embedding_20_embeddings_v:	?@@c
Qassignvariableop_58_adam_token_and_position_embedding_9_embedding_21_embeddings_v:d@g
Uassignvariableop_59_adam_transformer_block_8_multi_head_attention_8_dense_96_kernel_v:@@a
Sassignvariableop_60_adam_transformer_block_8_multi_head_attention_8_dense_96_bias_v:@g
Uassignvariableop_61_adam_transformer_block_8_multi_head_attention_8_dense_97_kernel_v:@@a
Sassignvariableop_62_adam_transformer_block_8_multi_head_attention_8_dense_97_bias_v:@g
Uassignvariableop_63_adam_transformer_block_8_multi_head_attention_8_dense_98_kernel_v:@@a
Sassignvariableop_64_adam_transformer_block_8_multi_head_attention_8_dense_98_bias_v:@g
Uassignvariableop_65_adam_transformer_block_8_multi_head_attention_8_dense_99_kernel_v:@@a
Sassignvariableop_66_adam_transformer_block_8_multi_head_attention_8_dense_99_bias_v:@=
+assignvariableop_67_adam_dense_100_kernel_v:@ 7
)assignvariableop_68_adam_dense_100_bias_v: =
+assignvariableop_69_adam_dense_101_kernel_v: @7
)assignvariableop_70_adam_dense_101_bias_v:@Y
Kassignvariableop_71_adam_transformer_block_8_layer_normalization_26_gamma_v:@X
Jassignvariableop_72_adam_transformer_block_8_layer_normalization_26_beta_v:@Y
Kassignvariableop_73_adam_transformer_block_8_layer_normalization_27_gamma_v:@X
Jassignvariableop_74_adam_transformer_block_8_layer_normalization_27_beta_v:@
identity_76??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_8?AssignVariableOp_9?(
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*?'
value?'B?'LB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*?
value?B?LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Z
dtypesP
N2L	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_dense_102_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_102_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_103_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_103_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpIassignvariableop_9_token_and_position_embedding_9_embedding_20_embeddingsIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpJassignvariableop_10_token_and_position_embedding_9_embedding_21_embeddingsIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpNassignvariableop_11_transformer_block_8_multi_head_attention_8_dense_96_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpLassignvariableop_12_transformer_block_8_multi_head_attention_8_dense_96_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpNassignvariableop_13_transformer_block_8_multi_head_attention_8_dense_97_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpLassignvariableop_14_transformer_block_8_multi_head_attention_8_dense_97_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpNassignvariableop_15_transformer_block_8_multi_head_attention_8_dense_98_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpLassignvariableop_16_transformer_block_8_multi_head_attention_8_dense_98_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpNassignvariableop_17_transformer_block_8_multi_head_attention_8_dense_99_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpLassignvariableop_18_transformer_block_8_multi_head_attention_8_dense_99_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp$assignvariableop_19_dense_100_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp"assignvariableop_20_dense_100_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp$assignvariableop_21_dense_101_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp"assignvariableop_22_dense_101_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpDassignvariableop_23_transformer_block_8_layer_normalization_26_gammaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpCassignvariableop_24_transformer_block_8_layer_normalization_26_betaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpDassignvariableop_25_transformer_block_8_layer_normalization_27_gammaIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOpCassignvariableop_26_transformer_block_8_layer_normalization_27_betaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOpassignvariableop_27_totalIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOpassignvariableop_28_countIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOpassignvariableop_29_total_1Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOpassignvariableop_30_count_1Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_dense_102_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_dense_102_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_dense_103_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_dense_103_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOpQassignvariableop_35_adam_token_and_position_embedding_9_embedding_20_embeddings_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOpQassignvariableop_36_adam_token_and_position_embedding_9_embedding_21_embeddings_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOpUassignvariableop_37_adam_transformer_block_8_multi_head_attention_8_dense_96_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOpSassignvariableop_38_adam_transformer_block_8_multi_head_attention_8_dense_96_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOpUassignvariableop_39_adam_transformer_block_8_multi_head_attention_8_dense_97_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOpSassignvariableop_40_adam_transformer_block_8_multi_head_attention_8_dense_97_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOpUassignvariableop_41_adam_transformer_block_8_multi_head_attention_8_dense_98_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOpSassignvariableop_42_adam_transformer_block_8_multi_head_attention_8_dense_98_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOpUassignvariableop_43_adam_transformer_block_8_multi_head_attention_8_dense_99_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOpSassignvariableop_44_adam_transformer_block_8_multi_head_attention_8_dense_99_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_dense_100_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_dense_100_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_dense_101_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_dense_101_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOpKassignvariableop_49_adam_transformer_block_8_layer_normalization_26_gamma_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOpJassignvariableop_50_adam_transformer_block_8_layer_normalization_26_beta_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOpKassignvariableop_51_adam_transformer_block_8_layer_normalization_27_gamma_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOpJassignvariableop_52_adam_transformer_block_8_layer_normalization_27_beta_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp+assignvariableop_53_adam_dense_102_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp)assignvariableop_54_adam_dense_102_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp+assignvariableop_55_adam_dense_103_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp)assignvariableop_56_adam_dense_103_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOpQassignvariableop_57_adam_token_and_position_embedding_9_embedding_20_embeddings_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOpQassignvariableop_58_adam_token_and_position_embedding_9_embedding_21_embeddings_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOpUassignvariableop_59_adam_transformer_block_8_multi_head_attention_8_dense_96_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOpSassignvariableop_60_adam_transformer_block_8_multi_head_attention_8_dense_96_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOpUassignvariableop_61_adam_transformer_block_8_multi_head_attention_8_dense_97_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOpSassignvariableop_62_adam_transformer_block_8_multi_head_attention_8_dense_97_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOpUassignvariableop_63_adam_transformer_block_8_multi_head_attention_8_dense_98_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOpSassignvariableop_64_adam_transformer_block_8_multi_head_attention_8_dense_98_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOpUassignvariableop_65_adam_transformer_block_8_multi_head_attention_8_dense_99_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOpSassignvariableop_66_adam_transformer_block_8_multi_head_attention_8_dense_99_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp+assignvariableop_67_adam_dense_100_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp)assignvariableop_68_adam_dense_100_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp+assignvariableop_69_adam_dense_101_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp)assignvariableop_70_adam_dense_101_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOpKassignvariableop_71_adam_transformer_block_8_layer_normalization_26_gamma_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOpJassignvariableop_72_adam_transformer_block_8_layer_normalization_26_beta_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOpKassignvariableop_73_adam_transformer_block_8_layer_normalization_27_gamma_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOpJassignvariableop_74_adam_transformer_block_8_layer_normalization_27_beta_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_749
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_75Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_75?
Identity_76IdentityIdentity_75:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_76"#
identity_76Identity_76:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
,__inference_sequential_8_layer_call_fn_93657

inputs
unknown:@ 
	unknown_0: 
	unknown_1: @
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_8_layer_call_and_return_conditional_losses_908922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????d@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????d@
 
_user_specified_nameinputs
?
?
Y__inference_token_and_position_embedding_9_layer_call_and_return_conditional_losses_91059
x5
#embedding_21_embedding_lookup_91046:d@6
#embedding_20_embedding_lookup_91052:	?@@
identity??embedding_20/embedding_lookup?embedding_21/embedding_lookup?
ShapeShapex*
T0*
_output_shapes
:2
Shape}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/delta?
rangeRangerange/start:output:0strided_slice:output:0range/delta:output:0*#
_output_shapes
:?????????2
range?
embedding_21/embedding_lookupResourceGather#embedding_21_embedding_lookup_91046range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding_21/embedding_lookup/91046*'
_output_shapes
:?????????@*
dtype02
embedding_21/embedding_lookup?
&embedding_21/embedding_lookup/IdentityIdentity&embedding_21/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding_21/embedding_lookup/91046*'
_output_shapes
:?????????@2(
&embedding_21/embedding_lookup/Identity?
(embedding_21/embedding_lookup/Identity_1Identity/embedding_21/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????@2*
(embedding_21/embedding_lookup/Identity_1r
embedding_20/CastCastx*

DstT0*

SrcT0*'
_output_shapes
:?????????d2
embedding_20/Cast?
embedding_20/embedding_lookupResourceGather#embedding_20_embedding_lookup_91052embedding_20/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding_20/embedding_lookup/91052*+
_output_shapes
:?????????d@*
dtype02
embedding_20/embedding_lookup?
&embedding_20/embedding_lookup/IdentityIdentity&embedding_20/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding_20/embedding_lookup/91052*+
_output_shapes
:?????????d@2(
&embedding_20/embedding_lookup/Identity?
(embedding_20/embedding_lookup/Identity_1Identity/embedding_20/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????d@2*
(embedding_20/embedding_lookup/Identity_1?
addAddV21embedding_20/embedding_lookup/Identity_1:output:01embedding_21/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:?????????d@2
add?
IdentityIdentityadd:z:0^embedding_20/embedding_lookup^embedding_21/embedding_lookup*
T0*+
_output_shapes
:?????????d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 2>
embedding_20/embedding_lookupembedding_20/embedding_lookup2>
embedding_21/embedding_lookupembedding_21/embedding_lookup:J F
'
_output_shapes
:?????????d

_user_specified_namex
?
?
)__inference_dense_102_layer_call_fn_93586

inputs
unknown:@
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_102_layer_call_and_return_conditional_losses_913682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
Y__inference_token_and_position_embedding_9_layer_call_and_return_conditional_losses_92952
x5
#embedding_21_embedding_lookup_92939:d@6
#embedding_20_embedding_lookup_92945:	?@@
identity??embedding_20/embedding_lookup?embedding_21/embedding_lookup?
ShapeShapex*
T0*
_output_shapes
:2
Shape}
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/delta?
rangeRangerange/start:output:0strided_slice:output:0range/delta:output:0*#
_output_shapes
:?????????2
range?
embedding_21/embedding_lookupResourceGather#embedding_21_embedding_lookup_92939range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding_21/embedding_lookup/92939*'
_output_shapes
:?????????@*
dtype02
embedding_21/embedding_lookup?
&embedding_21/embedding_lookup/IdentityIdentity&embedding_21/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding_21/embedding_lookup/92939*'
_output_shapes
:?????????@2(
&embedding_21/embedding_lookup/Identity?
(embedding_21/embedding_lookup/Identity_1Identity/embedding_21/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????@2*
(embedding_21/embedding_lookup/Identity_1r
embedding_20/CastCastx*

DstT0*

SrcT0*'
_output_shapes
:?????????d2
embedding_20/Cast?
embedding_20/embedding_lookupResourceGather#embedding_20_embedding_lookup_92945embedding_20/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*6
_class,
*(loc:@embedding_20/embedding_lookup/92945*+
_output_shapes
:?????????d@*
dtype02
embedding_20/embedding_lookup?
&embedding_20/embedding_lookup/IdentityIdentity&embedding_20/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*6
_class,
*(loc:@embedding_20/embedding_lookup/92945*+
_output_shapes
:?????????d@2(
&embedding_20/embedding_lookup/Identity?
(embedding_20/embedding_lookup/Identity_1Identity/embedding_20/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????d@2*
(embedding_20/embedding_lookup/Identity_1?
addAddV21embedding_20/embedding_lookup/Identity_1:output:01embedding_21/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:?????????d@2
add?
IdentityIdentityadd:z:0^embedding_20/embedding_lookup^embedding_21/embedding_lookup*
T0*+
_output_shapes
:?????????d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 2>
embedding_20/embedding_lookupembedding_20/embedding_lookup2>
embedding_21/embedding_lookupembedding_21/embedding_lookup:J F
'
_output_shapes
:?????????d

_user_specified_namex
?

?
D__inference_dense_102_layer_call_and_return_conditional_losses_91368

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
q
U__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_93544

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
V
:__inference_global_average_pooling1d_8_layer_call_fn_93533

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_910142
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
,__inference_sequential_8_layer_call_fn_93670

inputs
unknown:@ 
	unknown_0: 
	unknown_1: @
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_8_layer_call_and_return_conditional_losses_909522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????d@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????d@
 
_user_specified_nameinputs
?

?
D__inference_dense_103_layer_call_and_return_conditional_losses_93644

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
N__inference_transformer_block_8_layer_call_and_return_conditional_losses_91309

inputsS
Amulti_head_attention_8_dense_96_tensordot_readvariableop_resource:@@M
?multi_head_attention_8_dense_96_biasadd_readvariableop_resource:@S
Amulti_head_attention_8_dense_97_tensordot_readvariableop_resource:@@M
?multi_head_attention_8_dense_97_biasadd_readvariableop_resource:@S
Amulti_head_attention_8_dense_98_tensordot_readvariableop_resource:@@M
?multi_head_attention_8_dense_98_biasadd_readvariableop_resource:@S
Amulti_head_attention_8_dense_99_tensordot_readvariableop_resource:@@M
?multi_head_attention_8_dense_99_biasadd_readvariableop_resource:@J
<layer_normalization_26_batchnorm_mul_readvariableop_resource:@F
8layer_normalization_26_batchnorm_readvariableop_resource:@J
8sequential_8_dense_100_tensordot_readvariableop_resource:@ D
6sequential_8_dense_100_biasadd_readvariableop_resource: J
8sequential_8_dense_101_tensordot_readvariableop_resource: @D
6sequential_8_dense_101_biasadd_readvariableop_resource:@J
<layer_normalization_27_batchnorm_mul_readvariableop_resource:@F
8layer_normalization_27_batchnorm_readvariableop_resource:@
identity??/layer_normalization_26/batchnorm/ReadVariableOp?3layer_normalization_26/batchnorm/mul/ReadVariableOp?/layer_normalization_27/batchnorm/ReadVariableOp?3layer_normalization_27/batchnorm/mul/ReadVariableOp?6multi_head_attention_8/dense_96/BiasAdd/ReadVariableOp?8multi_head_attention_8/dense_96/Tensordot/ReadVariableOp?6multi_head_attention_8/dense_97/BiasAdd/ReadVariableOp?8multi_head_attention_8/dense_97/Tensordot/ReadVariableOp?6multi_head_attention_8/dense_98/BiasAdd/ReadVariableOp?8multi_head_attention_8/dense_98/Tensordot/ReadVariableOp?6multi_head_attention_8/dense_99/BiasAdd/ReadVariableOp?8multi_head_attention_8/dense_99/Tensordot/ReadVariableOp?-sequential_8/dense_100/BiasAdd/ReadVariableOp?/sequential_8/dense_100/Tensordot/ReadVariableOp?-sequential_8/dense_101/BiasAdd/ReadVariableOp?/sequential_8/dense_101/Tensordot/ReadVariableOpr
multi_head_attention_8/ShapeShapeinputs*
T0*
_output_shapes
:2
multi_head_attention_8/Shape?
*multi_head_attention_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*multi_head_attention_8/strided_slice/stack?
,multi_head_attention_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,multi_head_attention_8/strided_slice/stack_1?
,multi_head_attention_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,multi_head_attention_8/strided_slice/stack_2?
$multi_head_attention_8/strided_sliceStridedSlice%multi_head_attention_8/Shape:output:03multi_head_attention_8/strided_slice/stack:output:05multi_head_attention_8/strided_slice/stack_1:output:05multi_head_attention_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$multi_head_attention_8/strided_slice?
8multi_head_attention_8/dense_96/Tensordot/ReadVariableOpReadVariableOpAmulti_head_attention_8_dense_96_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02:
8multi_head_attention_8/dense_96/Tensordot/ReadVariableOp?
.multi_head_attention_8/dense_96/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:20
.multi_head_attention_8/dense_96/Tensordot/axes?
.multi_head_attention_8/dense_96/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       20
.multi_head_attention_8/dense_96/Tensordot/free?
/multi_head_attention_8/dense_96/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:21
/multi_head_attention_8/dense_96/Tensordot/Shape?
7multi_head_attention_8/dense_96/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_attention_8/dense_96/Tensordot/GatherV2/axis?
2multi_head_attention_8/dense_96/Tensordot/GatherV2GatherV28multi_head_attention_8/dense_96/Tensordot/Shape:output:07multi_head_attention_8/dense_96/Tensordot/free:output:0@multi_head_attention_8/dense_96/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:24
2multi_head_attention_8/dense_96/Tensordot/GatherV2?
9multi_head_attention_8/dense_96/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9multi_head_attention_8/dense_96/Tensordot/GatherV2_1/axis?
4multi_head_attention_8/dense_96/Tensordot/GatherV2_1GatherV28multi_head_attention_8/dense_96/Tensordot/Shape:output:07multi_head_attention_8/dense_96/Tensordot/axes:output:0Bmulti_head_attention_8/dense_96/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4multi_head_attention_8/dense_96/Tensordot/GatherV2_1?
/multi_head_attention_8/dense_96/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/multi_head_attention_8/dense_96/Tensordot/Const?
.multi_head_attention_8/dense_96/Tensordot/ProdProd;multi_head_attention_8/dense_96/Tensordot/GatherV2:output:08multi_head_attention_8/dense_96/Tensordot/Const:output:0*
T0*
_output_shapes
: 20
.multi_head_attention_8/dense_96/Tensordot/Prod?
1multi_head_attention_8/dense_96/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1multi_head_attention_8/dense_96/Tensordot/Const_1?
0multi_head_attention_8/dense_96/Tensordot/Prod_1Prod=multi_head_attention_8/dense_96/Tensordot/GatherV2_1:output:0:multi_head_attention_8/dense_96/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 22
0multi_head_attention_8/dense_96/Tensordot/Prod_1?
5multi_head_attention_8/dense_96/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5multi_head_attention_8/dense_96/Tensordot/concat/axis?
0multi_head_attention_8/dense_96/Tensordot/concatConcatV27multi_head_attention_8/dense_96/Tensordot/free:output:07multi_head_attention_8/dense_96/Tensordot/axes:output:0>multi_head_attention_8/dense_96/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:22
0multi_head_attention_8/dense_96/Tensordot/concat?
/multi_head_attention_8/dense_96/Tensordot/stackPack7multi_head_attention_8/dense_96/Tensordot/Prod:output:09multi_head_attention_8/dense_96/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:21
/multi_head_attention_8/dense_96/Tensordot/stack?
3multi_head_attention_8/dense_96/Tensordot/transpose	Transposeinputs9multi_head_attention_8/dense_96/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????d@25
3multi_head_attention_8/dense_96/Tensordot/transpose?
1multi_head_attention_8/dense_96/Tensordot/ReshapeReshape7multi_head_attention_8/dense_96/Tensordot/transpose:y:08multi_head_attention_8/dense_96/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????23
1multi_head_attention_8/dense_96/Tensordot/Reshape?
0multi_head_attention_8/dense_96/Tensordot/MatMulMatMul:multi_head_attention_8/dense_96/Tensordot/Reshape:output:0@multi_head_attention_8/dense_96/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@22
0multi_head_attention_8/dense_96/Tensordot/MatMul?
1multi_head_attention_8/dense_96/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@23
1multi_head_attention_8/dense_96/Tensordot/Const_2?
7multi_head_attention_8/dense_96/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_attention_8/dense_96/Tensordot/concat_1/axis?
2multi_head_attention_8/dense_96/Tensordot/concat_1ConcatV2;multi_head_attention_8/dense_96/Tensordot/GatherV2:output:0:multi_head_attention_8/dense_96/Tensordot/Const_2:output:0@multi_head_attention_8/dense_96/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:24
2multi_head_attention_8/dense_96/Tensordot/concat_1?
)multi_head_attention_8/dense_96/TensordotReshape:multi_head_attention_8/dense_96/Tensordot/MatMul:product:0;multi_head_attention_8/dense_96/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????d@2+
)multi_head_attention_8/dense_96/Tensordot?
6multi_head_attention_8/dense_96/BiasAdd/ReadVariableOpReadVariableOp?multi_head_attention_8_dense_96_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6multi_head_attention_8/dense_96/BiasAdd/ReadVariableOp?
'multi_head_attention_8/dense_96/BiasAddBiasAdd2multi_head_attention_8/dense_96/Tensordot:output:0>multi_head_attention_8/dense_96/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@2)
'multi_head_attention_8/dense_96/BiasAdd?
8multi_head_attention_8/dense_97/Tensordot/ReadVariableOpReadVariableOpAmulti_head_attention_8_dense_97_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02:
8multi_head_attention_8/dense_97/Tensordot/ReadVariableOp?
.multi_head_attention_8/dense_97/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:20
.multi_head_attention_8/dense_97/Tensordot/axes?
.multi_head_attention_8/dense_97/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       20
.multi_head_attention_8/dense_97/Tensordot/free?
/multi_head_attention_8/dense_97/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:21
/multi_head_attention_8/dense_97/Tensordot/Shape?
7multi_head_attention_8/dense_97/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_attention_8/dense_97/Tensordot/GatherV2/axis?
2multi_head_attention_8/dense_97/Tensordot/GatherV2GatherV28multi_head_attention_8/dense_97/Tensordot/Shape:output:07multi_head_attention_8/dense_97/Tensordot/free:output:0@multi_head_attention_8/dense_97/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:24
2multi_head_attention_8/dense_97/Tensordot/GatherV2?
9multi_head_attention_8/dense_97/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9multi_head_attention_8/dense_97/Tensordot/GatherV2_1/axis?
4multi_head_attention_8/dense_97/Tensordot/GatherV2_1GatherV28multi_head_attention_8/dense_97/Tensordot/Shape:output:07multi_head_attention_8/dense_97/Tensordot/axes:output:0Bmulti_head_attention_8/dense_97/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4multi_head_attention_8/dense_97/Tensordot/GatherV2_1?
/multi_head_attention_8/dense_97/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/multi_head_attention_8/dense_97/Tensordot/Const?
.multi_head_attention_8/dense_97/Tensordot/ProdProd;multi_head_attention_8/dense_97/Tensordot/GatherV2:output:08multi_head_attention_8/dense_97/Tensordot/Const:output:0*
T0*
_output_shapes
: 20
.multi_head_attention_8/dense_97/Tensordot/Prod?
1multi_head_attention_8/dense_97/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1multi_head_attention_8/dense_97/Tensordot/Const_1?
0multi_head_attention_8/dense_97/Tensordot/Prod_1Prod=multi_head_attention_8/dense_97/Tensordot/GatherV2_1:output:0:multi_head_attention_8/dense_97/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 22
0multi_head_attention_8/dense_97/Tensordot/Prod_1?
5multi_head_attention_8/dense_97/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5multi_head_attention_8/dense_97/Tensordot/concat/axis?
0multi_head_attention_8/dense_97/Tensordot/concatConcatV27multi_head_attention_8/dense_97/Tensordot/free:output:07multi_head_attention_8/dense_97/Tensordot/axes:output:0>multi_head_attention_8/dense_97/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:22
0multi_head_attention_8/dense_97/Tensordot/concat?
/multi_head_attention_8/dense_97/Tensordot/stackPack7multi_head_attention_8/dense_97/Tensordot/Prod:output:09multi_head_attention_8/dense_97/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:21
/multi_head_attention_8/dense_97/Tensordot/stack?
3multi_head_attention_8/dense_97/Tensordot/transpose	Transposeinputs9multi_head_attention_8/dense_97/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????d@25
3multi_head_attention_8/dense_97/Tensordot/transpose?
1multi_head_attention_8/dense_97/Tensordot/ReshapeReshape7multi_head_attention_8/dense_97/Tensordot/transpose:y:08multi_head_attention_8/dense_97/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????23
1multi_head_attention_8/dense_97/Tensordot/Reshape?
0multi_head_attention_8/dense_97/Tensordot/MatMulMatMul:multi_head_attention_8/dense_97/Tensordot/Reshape:output:0@multi_head_attention_8/dense_97/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@22
0multi_head_attention_8/dense_97/Tensordot/MatMul?
1multi_head_attention_8/dense_97/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@23
1multi_head_attention_8/dense_97/Tensordot/Const_2?
7multi_head_attention_8/dense_97/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_attention_8/dense_97/Tensordot/concat_1/axis?
2multi_head_attention_8/dense_97/Tensordot/concat_1ConcatV2;multi_head_attention_8/dense_97/Tensordot/GatherV2:output:0:multi_head_attention_8/dense_97/Tensordot/Const_2:output:0@multi_head_attention_8/dense_97/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:24
2multi_head_attention_8/dense_97/Tensordot/concat_1?
)multi_head_attention_8/dense_97/TensordotReshape:multi_head_attention_8/dense_97/Tensordot/MatMul:product:0;multi_head_attention_8/dense_97/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????d@2+
)multi_head_attention_8/dense_97/Tensordot?
6multi_head_attention_8/dense_97/BiasAdd/ReadVariableOpReadVariableOp?multi_head_attention_8_dense_97_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6multi_head_attention_8/dense_97/BiasAdd/ReadVariableOp?
'multi_head_attention_8/dense_97/BiasAddBiasAdd2multi_head_attention_8/dense_97/Tensordot:output:0>multi_head_attention_8/dense_97/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@2)
'multi_head_attention_8/dense_97/BiasAdd?
8multi_head_attention_8/dense_98/Tensordot/ReadVariableOpReadVariableOpAmulti_head_attention_8_dense_98_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02:
8multi_head_attention_8/dense_98/Tensordot/ReadVariableOp?
.multi_head_attention_8/dense_98/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:20
.multi_head_attention_8/dense_98/Tensordot/axes?
.multi_head_attention_8/dense_98/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       20
.multi_head_attention_8/dense_98/Tensordot/free?
/multi_head_attention_8/dense_98/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:21
/multi_head_attention_8/dense_98/Tensordot/Shape?
7multi_head_attention_8/dense_98/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_attention_8/dense_98/Tensordot/GatherV2/axis?
2multi_head_attention_8/dense_98/Tensordot/GatherV2GatherV28multi_head_attention_8/dense_98/Tensordot/Shape:output:07multi_head_attention_8/dense_98/Tensordot/free:output:0@multi_head_attention_8/dense_98/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:24
2multi_head_attention_8/dense_98/Tensordot/GatherV2?
9multi_head_attention_8/dense_98/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9multi_head_attention_8/dense_98/Tensordot/GatherV2_1/axis?
4multi_head_attention_8/dense_98/Tensordot/GatherV2_1GatherV28multi_head_attention_8/dense_98/Tensordot/Shape:output:07multi_head_attention_8/dense_98/Tensordot/axes:output:0Bmulti_head_attention_8/dense_98/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4multi_head_attention_8/dense_98/Tensordot/GatherV2_1?
/multi_head_attention_8/dense_98/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/multi_head_attention_8/dense_98/Tensordot/Const?
.multi_head_attention_8/dense_98/Tensordot/ProdProd;multi_head_attention_8/dense_98/Tensordot/GatherV2:output:08multi_head_attention_8/dense_98/Tensordot/Const:output:0*
T0*
_output_shapes
: 20
.multi_head_attention_8/dense_98/Tensordot/Prod?
1multi_head_attention_8/dense_98/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1multi_head_attention_8/dense_98/Tensordot/Const_1?
0multi_head_attention_8/dense_98/Tensordot/Prod_1Prod=multi_head_attention_8/dense_98/Tensordot/GatherV2_1:output:0:multi_head_attention_8/dense_98/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 22
0multi_head_attention_8/dense_98/Tensordot/Prod_1?
5multi_head_attention_8/dense_98/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5multi_head_attention_8/dense_98/Tensordot/concat/axis?
0multi_head_attention_8/dense_98/Tensordot/concatConcatV27multi_head_attention_8/dense_98/Tensordot/free:output:07multi_head_attention_8/dense_98/Tensordot/axes:output:0>multi_head_attention_8/dense_98/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:22
0multi_head_attention_8/dense_98/Tensordot/concat?
/multi_head_attention_8/dense_98/Tensordot/stackPack7multi_head_attention_8/dense_98/Tensordot/Prod:output:09multi_head_attention_8/dense_98/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:21
/multi_head_attention_8/dense_98/Tensordot/stack?
3multi_head_attention_8/dense_98/Tensordot/transpose	Transposeinputs9multi_head_attention_8/dense_98/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????d@25
3multi_head_attention_8/dense_98/Tensordot/transpose?
1multi_head_attention_8/dense_98/Tensordot/ReshapeReshape7multi_head_attention_8/dense_98/Tensordot/transpose:y:08multi_head_attention_8/dense_98/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????23
1multi_head_attention_8/dense_98/Tensordot/Reshape?
0multi_head_attention_8/dense_98/Tensordot/MatMulMatMul:multi_head_attention_8/dense_98/Tensordot/Reshape:output:0@multi_head_attention_8/dense_98/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@22
0multi_head_attention_8/dense_98/Tensordot/MatMul?
1multi_head_attention_8/dense_98/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@23
1multi_head_attention_8/dense_98/Tensordot/Const_2?
7multi_head_attention_8/dense_98/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_attention_8/dense_98/Tensordot/concat_1/axis?
2multi_head_attention_8/dense_98/Tensordot/concat_1ConcatV2;multi_head_attention_8/dense_98/Tensordot/GatherV2:output:0:multi_head_attention_8/dense_98/Tensordot/Const_2:output:0@multi_head_attention_8/dense_98/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:24
2multi_head_attention_8/dense_98/Tensordot/concat_1?
)multi_head_attention_8/dense_98/TensordotReshape:multi_head_attention_8/dense_98/Tensordot/MatMul:product:0;multi_head_attention_8/dense_98/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????d@2+
)multi_head_attention_8/dense_98/Tensordot?
6multi_head_attention_8/dense_98/BiasAdd/ReadVariableOpReadVariableOp?multi_head_attention_8_dense_98_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6multi_head_attention_8/dense_98/BiasAdd/ReadVariableOp?
'multi_head_attention_8/dense_98/BiasAddBiasAdd2multi_head_attention_8/dense_98/Tensordot:output:0>multi_head_attention_8/dense_98/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@2)
'multi_head_attention_8/dense_98/BiasAdd?
&multi_head_attention_8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&multi_head_attention_8/Reshape/shape/1?
&multi_head_attention_8/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2(
&multi_head_attention_8/Reshape/shape/2?
&multi_head_attention_8/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2(
&multi_head_attention_8/Reshape/shape/3?
$multi_head_attention_8/Reshape/shapePack-multi_head_attention_8/strided_slice:output:0/multi_head_attention_8/Reshape/shape/1:output:0/multi_head_attention_8/Reshape/shape/2:output:0/multi_head_attention_8/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2&
$multi_head_attention_8/Reshape/shape?
multi_head_attention_8/ReshapeReshape0multi_head_attention_8/dense_96/BiasAdd:output:0-multi_head_attention_8/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2 
multi_head_attention_8/Reshape?
%multi_head_attention_8/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2'
%multi_head_attention_8/transpose/perm?
 multi_head_attention_8/transpose	Transpose'multi_head_attention_8/Reshape:output:0.multi_head_attention_8/transpose/perm:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2"
 multi_head_attention_8/transpose?
(multi_head_attention_8/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(multi_head_attention_8/Reshape_1/shape/1?
(multi_head_attention_8/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(multi_head_attention_8/Reshape_1/shape/2?
(multi_head_attention_8/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2*
(multi_head_attention_8/Reshape_1/shape/3?
&multi_head_attention_8/Reshape_1/shapePack-multi_head_attention_8/strided_slice:output:01multi_head_attention_8/Reshape_1/shape/1:output:01multi_head_attention_8/Reshape_1/shape/2:output:01multi_head_attention_8/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_attention_8/Reshape_1/shape?
 multi_head_attention_8/Reshape_1Reshape0multi_head_attention_8/dense_97/BiasAdd:output:0/multi_head_attention_8/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2"
 multi_head_attention_8/Reshape_1?
'multi_head_attention_8/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2)
'multi_head_attention_8/transpose_1/perm?
"multi_head_attention_8/transpose_1	Transpose)multi_head_attention_8/Reshape_1:output:00multi_head_attention_8/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2$
"multi_head_attention_8/transpose_1?
(multi_head_attention_8/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(multi_head_attention_8/Reshape_2/shape/1?
(multi_head_attention_8/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(multi_head_attention_8/Reshape_2/shape/2?
(multi_head_attention_8/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2*
(multi_head_attention_8/Reshape_2/shape/3?
&multi_head_attention_8/Reshape_2/shapePack-multi_head_attention_8/strided_slice:output:01multi_head_attention_8/Reshape_2/shape/1:output:01multi_head_attention_8/Reshape_2/shape/2:output:01multi_head_attention_8/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_attention_8/Reshape_2/shape?
 multi_head_attention_8/Reshape_2Reshape0multi_head_attention_8/dense_98/BiasAdd:output:0/multi_head_attention_8/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2"
 multi_head_attention_8/Reshape_2?
'multi_head_attention_8/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2)
'multi_head_attention_8/transpose_2/perm?
"multi_head_attention_8/transpose_2	Transpose)multi_head_attention_8/Reshape_2:output:00multi_head_attention_8/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2$
"multi_head_attention_8/transpose_2?
multi_head_attention_8/MatMulBatchMatMulV2$multi_head_attention_8/transpose:y:0&multi_head_attention_8/transpose_1:y:0*
T0*A
_output_shapes/
-:+???????????????????????????*
adj_y(2
multi_head_attention_8/MatMul?
multi_head_attention_8/Shape_1Shape&multi_head_attention_8/transpose_1:y:0*
T0*
_output_shapes
:2 
multi_head_attention_8/Shape_1?
,multi_head_attention_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2.
,multi_head_attention_8/strided_slice_1/stack?
.multi_head_attention_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 20
.multi_head_attention_8/strided_slice_1/stack_1?
.multi_head_attention_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.multi_head_attention_8/strided_slice_1/stack_2?
&multi_head_attention_8/strided_slice_1StridedSlice'multi_head_attention_8/Shape_1:output:05multi_head_attention_8/strided_slice_1/stack:output:07multi_head_attention_8/strided_slice_1/stack_1:output:07multi_head_attention_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&multi_head_attention_8/strided_slice_1?
multi_head_attention_8/CastCast/multi_head_attention_8/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
multi_head_attention_8/Cast?
multi_head_attention_8/SqrtSqrtmulti_head_attention_8/Cast:y:0*
T0*
_output_shapes
: 2
multi_head_attention_8/Sqrt?
multi_head_attention_8/truedivRealDiv&multi_head_attention_8/MatMul:output:0multi_head_attention_8/Sqrt:y:0*
T0*A
_output_shapes/
-:+???????????????????????????2 
multi_head_attention_8/truediv?
multi_head_attention_8/SoftmaxSoftmax"multi_head_attention_8/truediv:z:0*
T0*A
_output_shapes/
-:+???????????????????????????2 
multi_head_attention_8/Softmax?
multi_head_attention_8/MatMul_1BatchMatMulV2(multi_head_attention_8/Softmax:softmax:0&multi_head_attention_8/transpose_2:y:0*
T0*8
_output_shapes&
$:"?????????????????? 2!
multi_head_attention_8/MatMul_1?
'multi_head_attention_8/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             2)
'multi_head_attention_8/transpose_3/perm?
"multi_head_attention_8/transpose_3	Transpose(multi_head_attention_8/MatMul_1:output:00multi_head_attention_8/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2$
"multi_head_attention_8/transpose_3?
(multi_head_attention_8/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(multi_head_attention_8/Reshape_3/shape/1?
(multi_head_attention_8/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@2*
(multi_head_attention_8/Reshape_3/shape/2?
&multi_head_attention_8/Reshape_3/shapePack-multi_head_attention_8/strided_slice:output:01multi_head_attention_8/Reshape_3/shape/1:output:01multi_head_attention_8/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2(
&multi_head_attention_8/Reshape_3/shape?
 multi_head_attention_8/Reshape_3Reshape&multi_head_attention_8/transpose_3:y:0/multi_head_attention_8/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :??????????????????@2"
 multi_head_attention_8/Reshape_3?
8multi_head_attention_8/dense_99/Tensordot/ReadVariableOpReadVariableOpAmulti_head_attention_8_dense_99_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02:
8multi_head_attention_8/dense_99/Tensordot/ReadVariableOp?
.multi_head_attention_8/dense_99/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:20
.multi_head_attention_8/dense_99/Tensordot/axes?
.multi_head_attention_8/dense_99/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       20
.multi_head_attention_8/dense_99/Tensordot/free?
/multi_head_attention_8/dense_99/Tensordot/ShapeShape)multi_head_attention_8/Reshape_3:output:0*
T0*
_output_shapes
:21
/multi_head_attention_8/dense_99/Tensordot/Shape?
7multi_head_attention_8/dense_99/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_attention_8/dense_99/Tensordot/GatherV2/axis?
2multi_head_attention_8/dense_99/Tensordot/GatherV2GatherV28multi_head_attention_8/dense_99/Tensordot/Shape:output:07multi_head_attention_8/dense_99/Tensordot/free:output:0@multi_head_attention_8/dense_99/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:24
2multi_head_attention_8/dense_99/Tensordot/GatherV2?
9multi_head_attention_8/dense_99/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9multi_head_attention_8/dense_99/Tensordot/GatherV2_1/axis?
4multi_head_attention_8/dense_99/Tensordot/GatherV2_1GatherV28multi_head_attention_8/dense_99/Tensordot/Shape:output:07multi_head_attention_8/dense_99/Tensordot/axes:output:0Bmulti_head_attention_8/dense_99/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4multi_head_attention_8/dense_99/Tensordot/GatherV2_1?
/multi_head_attention_8/dense_99/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/multi_head_attention_8/dense_99/Tensordot/Const?
.multi_head_attention_8/dense_99/Tensordot/ProdProd;multi_head_attention_8/dense_99/Tensordot/GatherV2:output:08multi_head_attention_8/dense_99/Tensordot/Const:output:0*
T0*
_output_shapes
: 20
.multi_head_attention_8/dense_99/Tensordot/Prod?
1multi_head_attention_8/dense_99/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1multi_head_attention_8/dense_99/Tensordot/Const_1?
0multi_head_attention_8/dense_99/Tensordot/Prod_1Prod=multi_head_attention_8/dense_99/Tensordot/GatherV2_1:output:0:multi_head_attention_8/dense_99/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 22
0multi_head_attention_8/dense_99/Tensordot/Prod_1?
5multi_head_attention_8/dense_99/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5multi_head_attention_8/dense_99/Tensordot/concat/axis?
0multi_head_attention_8/dense_99/Tensordot/concatConcatV27multi_head_attention_8/dense_99/Tensordot/free:output:07multi_head_attention_8/dense_99/Tensordot/axes:output:0>multi_head_attention_8/dense_99/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:22
0multi_head_attention_8/dense_99/Tensordot/concat?
/multi_head_attention_8/dense_99/Tensordot/stackPack7multi_head_attention_8/dense_99/Tensordot/Prod:output:09multi_head_attention_8/dense_99/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:21
/multi_head_attention_8/dense_99/Tensordot/stack?
3multi_head_attention_8/dense_99/Tensordot/transpose	Transpose)multi_head_attention_8/Reshape_3:output:09multi_head_attention_8/dense_99/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :??????????????????@25
3multi_head_attention_8/dense_99/Tensordot/transpose?
1multi_head_attention_8/dense_99/Tensordot/ReshapeReshape7multi_head_attention_8/dense_99/Tensordot/transpose:y:08multi_head_attention_8/dense_99/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????23
1multi_head_attention_8/dense_99/Tensordot/Reshape?
0multi_head_attention_8/dense_99/Tensordot/MatMulMatMul:multi_head_attention_8/dense_99/Tensordot/Reshape:output:0@multi_head_attention_8/dense_99/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@22
0multi_head_attention_8/dense_99/Tensordot/MatMul?
1multi_head_attention_8/dense_99/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@23
1multi_head_attention_8/dense_99/Tensordot/Const_2?
7multi_head_attention_8/dense_99/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_attention_8/dense_99/Tensordot/concat_1/axis?
2multi_head_attention_8/dense_99/Tensordot/concat_1ConcatV2;multi_head_attention_8/dense_99/Tensordot/GatherV2:output:0:multi_head_attention_8/dense_99/Tensordot/Const_2:output:0@multi_head_attention_8/dense_99/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:24
2multi_head_attention_8/dense_99/Tensordot/concat_1?
)multi_head_attention_8/dense_99/TensordotReshape:multi_head_attention_8/dense_99/Tensordot/MatMul:product:0;multi_head_attention_8/dense_99/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :??????????????????@2+
)multi_head_attention_8/dense_99/Tensordot?
6multi_head_attention_8/dense_99/BiasAdd/ReadVariableOpReadVariableOp?multi_head_attention_8_dense_99_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6multi_head_attention_8/dense_99/BiasAdd/ReadVariableOp?
'multi_head_attention_8/dense_99/BiasAddBiasAdd2multi_head_attention_8/dense_99/Tensordot:output:0>multi_head_attention_8/dense_99/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????@2)
'multi_head_attention_8/dense_99/BiasAdd?
dropout_42/IdentityIdentity0multi_head_attention_8/dense_99/BiasAdd:output:0*
T0*4
_output_shapes"
 :??????????????????@2
dropout_42/Identityo
addAddV2inputsdropout_42/Identity:output:0*
T0*+
_output_shapes
:?????????d@2
add?
5layer_normalization_26/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_26/moments/mean/reduction_indices?
#layer_normalization_26/moments/meanMeanadd:z:0>layer_normalization_26/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????d*
	keep_dims(2%
#layer_normalization_26/moments/mean?
+layer_normalization_26/moments/StopGradientStopGradient,layer_normalization_26/moments/mean:output:0*
T0*+
_output_shapes
:?????????d2-
+layer_normalization_26/moments/StopGradient?
0layer_normalization_26/moments/SquaredDifferenceSquaredDifferenceadd:z:04layer_normalization_26/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????d@22
0layer_normalization_26/moments/SquaredDifference?
9layer_normalization_26/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_26/moments/variance/reduction_indices?
'layer_normalization_26/moments/varianceMean4layer_normalization_26/moments/SquaredDifference:z:0Blayer_normalization_26/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????d*
	keep_dims(2)
'layer_normalization_26/moments/variance?
&layer_normalization_26/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52(
&layer_normalization_26/batchnorm/add/y?
$layer_normalization_26/batchnorm/addAddV20layer_normalization_26/moments/variance:output:0/layer_normalization_26/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????d2&
$layer_normalization_26/batchnorm/add?
&layer_normalization_26/batchnorm/RsqrtRsqrt(layer_normalization_26/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????d2(
&layer_normalization_26/batchnorm/Rsqrt?
3layer_normalization_26/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_26_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype025
3layer_normalization_26/batchnorm/mul/ReadVariableOp?
$layer_normalization_26/batchnorm/mulMul*layer_normalization_26/batchnorm/Rsqrt:y:0;layer_normalization_26/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@2&
$layer_normalization_26/batchnorm/mul?
&layer_normalization_26/batchnorm/mul_1Muladd:z:0(layer_normalization_26/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????d@2(
&layer_normalization_26/batchnorm/mul_1?
&layer_normalization_26/batchnorm/mul_2Mul,layer_normalization_26/moments/mean:output:0(layer_normalization_26/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????d@2(
&layer_normalization_26/batchnorm/mul_2?
/layer_normalization_26/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_26_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype021
/layer_normalization_26/batchnorm/ReadVariableOp?
$layer_normalization_26/batchnorm/subSub7layer_normalization_26/batchnorm/ReadVariableOp:value:0*layer_normalization_26/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????d@2&
$layer_normalization_26/batchnorm/sub?
&layer_normalization_26/batchnorm/add_1AddV2*layer_normalization_26/batchnorm/mul_1:z:0(layer_normalization_26/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????d@2(
&layer_normalization_26/batchnorm/add_1?
/sequential_8/dense_100/Tensordot/ReadVariableOpReadVariableOp8sequential_8_dense_100_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype021
/sequential_8/dense_100/Tensordot/ReadVariableOp?
%sequential_8/dense_100/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_8/dense_100/Tensordot/axes?
%sequential_8/dense_100/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_8/dense_100/Tensordot/free?
&sequential_8/dense_100/Tensordot/ShapeShape*layer_normalization_26/batchnorm/add_1:z:0*
T0*
_output_shapes
:2(
&sequential_8/dense_100/Tensordot/Shape?
.sequential_8/dense_100/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_8/dense_100/Tensordot/GatherV2/axis?
)sequential_8/dense_100/Tensordot/GatherV2GatherV2/sequential_8/dense_100/Tensordot/Shape:output:0.sequential_8/dense_100/Tensordot/free:output:07sequential_8/dense_100/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_8/dense_100/Tensordot/GatherV2?
0sequential_8/dense_100/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_8/dense_100/Tensordot/GatherV2_1/axis?
+sequential_8/dense_100/Tensordot/GatherV2_1GatherV2/sequential_8/dense_100/Tensordot/Shape:output:0.sequential_8/dense_100/Tensordot/axes:output:09sequential_8/dense_100/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_8/dense_100/Tensordot/GatherV2_1?
&sequential_8/dense_100/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_8/dense_100/Tensordot/Const?
%sequential_8/dense_100/Tensordot/ProdProd2sequential_8/dense_100/Tensordot/GatherV2:output:0/sequential_8/dense_100/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_8/dense_100/Tensordot/Prod?
(sequential_8/dense_100/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_8/dense_100/Tensordot/Const_1?
'sequential_8/dense_100/Tensordot/Prod_1Prod4sequential_8/dense_100/Tensordot/GatherV2_1:output:01sequential_8/dense_100/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_8/dense_100/Tensordot/Prod_1?
,sequential_8/dense_100/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_8/dense_100/Tensordot/concat/axis?
'sequential_8/dense_100/Tensordot/concatConcatV2.sequential_8/dense_100/Tensordot/free:output:0.sequential_8/dense_100/Tensordot/axes:output:05sequential_8/dense_100/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_8/dense_100/Tensordot/concat?
&sequential_8/dense_100/Tensordot/stackPack.sequential_8/dense_100/Tensordot/Prod:output:00sequential_8/dense_100/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_8/dense_100/Tensordot/stack?
*sequential_8/dense_100/Tensordot/transpose	Transpose*layer_normalization_26/batchnorm/add_1:z:00sequential_8/dense_100/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????d@2,
*sequential_8/dense_100/Tensordot/transpose?
(sequential_8/dense_100/Tensordot/ReshapeReshape.sequential_8/dense_100/Tensordot/transpose:y:0/sequential_8/dense_100/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(sequential_8/dense_100/Tensordot/Reshape?
'sequential_8/dense_100/Tensordot/MatMulMatMul1sequential_8/dense_100/Tensordot/Reshape:output:07sequential_8/dense_100/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2)
'sequential_8/dense_100/Tensordot/MatMul?
(sequential_8/dense_100/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_8/dense_100/Tensordot/Const_2?
.sequential_8/dense_100/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_8/dense_100/Tensordot/concat_1/axis?
)sequential_8/dense_100/Tensordot/concat_1ConcatV22sequential_8/dense_100/Tensordot/GatherV2:output:01sequential_8/dense_100/Tensordot/Const_2:output:07sequential_8/dense_100/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_8/dense_100/Tensordot/concat_1?
 sequential_8/dense_100/TensordotReshape1sequential_8/dense_100/Tensordot/MatMul:product:02sequential_8/dense_100/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????d 2"
 sequential_8/dense_100/Tensordot?
-sequential_8/dense_100/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_dense_100_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_8/dense_100/BiasAdd/ReadVariableOp?
sequential_8/dense_100/BiasAddBiasAdd)sequential_8/dense_100/Tensordot:output:05sequential_8/dense_100/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d 2 
sequential_8/dense_100/BiasAdd?
sequential_8/dense_100/ReluRelu'sequential_8/dense_100/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d 2
sequential_8/dense_100/Relu?
/sequential_8/dense_101/Tensordot/ReadVariableOpReadVariableOp8sequential_8_dense_101_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype021
/sequential_8/dense_101/Tensordot/ReadVariableOp?
%sequential_8/dense_101/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_8/dense_101/Tensordot/axes?
%sequential_8/dense_101/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_8/dense_101/Tensordot/free?
&sequential_8/dense_101/Tensordot/ShapeShape)sequential_8/dense_100/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_8/dense_101/Tensordot/Shape?
.sequential_8/dense_101/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_8/dense_101/Tensordot/GatherV2/axis?
)sequential_8/dense_101/Tensordot/GatherV2GatherV2/sequential_8/dense_101/Tensordot/Shape:output:0.sequential_8/dense_101/Tensordot/free:output:07sequential_8/dense_101/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_8/dense_101/Tensordot/GatherV2?
0sequential_8/dense_101/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_8/dense_101/Tensordot/GatherV2_1/axis?
+sequential_8/dense_101/Tensordot/GatherV2_1GatherV2/sequential_8/dense_101/Tensordot/Shape:output:0.sequential_8/dense_101/Tensordot/axes:output:09sequential_8/dense_101/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_8/dense_101/Tensordot/GatherV2_1?
&sequential_8/dense_101/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_8/dense_101/Tensordot/Const?
%sequential_8/dense_101/Tensordot/ProdProd2sequential_8/dense_101/Tensordot/GatherV2:output:0/sequential_8/dense_101/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_8/dense_101/Tensordot/Prod?
(sequential_8/dense_101/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_8/dense_101/Tensordot/Const_1?
'sequential_8/dense_101/Tensordot/Prod_1Prod4sequential_8/dense_101/Tensordot/GatherV2_1:output:01sequential_8/dense_101/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_8/dense_101/Tensordot/Prod_1?
,sequential_8/dense_101/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_8/dense_101/Tensordot/concat/axis?
'sequential_8/dense_101/Tensordot/concatConcatV2.sequential_8/dense_101/Tensordot/free:output:0.sequential_8/dense_101/Tensordot/axes:output:05sequential_8/dense_101/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_8/dense_101/Tensordot/concat?
&sequential_8/dense_101/Tensordot/stackPack.sequential_8/dense_101/Tensordot/Prod:output:00sequential_8/dense_101/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_8/dense_101/Tensordot/stack?
*sequential_8/dense_101/Tensordot/transpose	Transpose)sequential_8/dense_100/Relu:activations:00sequential_8/dense_101/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????d 2,
*sequential_8/dense_101/Tensordot/transpose?
(sequential_8/dense_101/Tensordot/ReshapeReshape.sequential_8/dense_101/Tensordot/transpose:y:0/sequential_8/dense_101/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(sequential_8/dense_101/Tensordot/Reshape?
'sequential_8/dense_101/Tensordot/MatMulMatMul1sequential_8/dense_101/Tensordot/Reshape:output:07sequential_8/dense_101/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2)
'sequential_8/dense_101/Tensordot/MatMul?
(sequential_8/dense_101/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2*
(sequential_8/dense_101/Tensordot/Const_2?
.sequential_8/dense_101/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_8/dense_101/Tensordot/concat_1/axis?
)sequential_8/dense_101/Tensordot/concat_1ConcatV22sequential_8/dense_101/Tensordot/GatherV2:output:01sequential_8/dense_101/Tensordot/Const_2:output:07sequential_8/dense_101/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_8/dense_101/Tensordot/concat_1?
 sequential_8/dense_101/TensordotReshape1sequential_8/dense_101/Tensordot/MatMul:product:02sequential_8/dense_101/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????d@2"
 sequential_8/dense_101/Tensordot?
-sequential_8/dense_101/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_dense_101_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-sequential_8/dense_101/BiasAdd/ReadVariableOp?
sequential_8/dense_101/BiasAddBiasAdd)sequential_8/dense_101/Tensordot:output:05sequential_8/dense_101/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@2 
sequential_8/dense_101/BiasAdd?
dropout_43/IdentityIdentity'sequential_8/dense_101/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d@2
dropout_43/Identity?
add_1AddV2*layer_normalization_26/batchnorm/add_1:z:0dropout_43/Identity:output:0*
T0*+
_output_shapes
:?????????d@2
add_1?
5layer_normalization_27/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:27
5layer_normalization_27/moments/mean/reduction_indices?
#layer_normalization_27/moments/meanMean	add_1:z:0>layer_normalization_27/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????d*
	keep_dims(2%
#layer_normalization_27/moments/mean?
+layer_normalization_27/moments/StopGradientStopGradient,layer_normalization_27/moments/mean:output:0*
T0*+
_output_shapes
:?????????d2-
+layer_normalization_27/moments/StopGradient?
0layer_normalization_27/moments/SquaredDifferenceSquaredDifference	add_1:z:04layer_normalization_27/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????d@22
0layer_normalization_27/moments/SquaredDifference?
9layer_normalization_27/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9layer_normalization_27/moments/variance/reduction_indices?
'layer_normalization_27/moments/varianceMean4layer_normalization_27/moments/SquaredDifference:z:0Blayer_normalization_27/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????d*
	keep_dims(2)
'layer_normalization_27/moments/variance?
&layer_normalization_27/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52(
&layer_normalization_27/batchnorm/add/y?
$layer_normalization_27/batchnorm/addAddV20layer_normalization_27/moments/variance:output:0/layer_normalization_27/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????d2&
$layer_normalization_27/batchnorm/add?
&layer_normalization_27/batchnorm/RsqrtRsqrt(layer_normalization_27/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????d2(
&layer_normalization_27/batchnorm/Rsqrt?
3layer_normalization_27/batchnorm/mul/ReadVariableOpReadVariableOp<layer_normalization_27_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype025
3layer_normalization_27/batchnorm/mul/ReadVariableOp?
$layer_normalization_27/batchnorm/mulMul*layer_normalization_27/batchnorm/Rsqrt:y:0;layer_normalization_27/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@2&
$layer_normalization_27/batchnorm/mul?
&layer_normalization_27/batchnorm/mul_1Mul	add_1:z:0(layer_normalization_27/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????d@2(
&layer_normalization_27/batchnorm/mul_1?
&layer_normalization_27/batchnorm/mul_2Mul,layer_normalization_27/moments/mean:output:0(layer_normalization_27/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????d@2(
&layer_normalization_27/batchnorm/mul_2?
/layer_normalization_27/batchnorm/ReadVariableOpReadVariableOp8layer_normalization_27_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype021
/layer_normalization_27/batchnorm/ReadVariableOp?
$layer_normalization_27/batchnorm/subSub7layer_normalization_27/batchnorm/ReadVariableOp:value:0*layer_normalization_27/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????d@2&
$layer_normalization_27/batchnorm/sub?
&layer_normalization_27/batchnorm/add_1AddV2*layer_normalization_27/batchnorm/mul_1:z:0(layer_normalization_27/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????d@2(
&layer_normalization_27/batchnorm/add_1?
IdentityIdentity*layer_normalization_27/batchnorm/add_1:z:00^layer_normalization_26/batchnorm/ReadVariableOp4^layer_normalization_26/batchnorm/mul/ReadVariableOp0^layer_normalization_27/batchnorm/ReadVariableOp4^layer_normalization_27/batchnorm/mul/ReadVariableOp7^multi_head_attention_8/dense_96/BiasAdd/ReadVariableOp9^multi_head_attention_8/dense_96/Tensordot/ReadVariableOp7^multi_head_attention_8/dense_97/BiasAdd/ReadVariableOp9^multi_head_attention_8/dense_97/Tensordot/ReadVariableOp7^multi_head_attention_8/dense_98/BiasAdd/ReadVariableOp9^multi_head_attention_8/dense_98/Tensordot/ReadVariableOp7^multi_head_attention_8/dense_99/BiasAdd/ReadVariableOp9^multi_head_attention_8/dense_99/Tensordot/ReadVariableOp.^sequential_8/dense_100/BiasAdd/ReadVariableOp0^sequential_8/dense_100/Tensordot/ReadVariableOp.^sequential_8/dense_101/BiasAdd/ReadVariableOp0^sequential_8/dense_101/Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????d@: : : : : : : : : : : : : : : : 2b
/layer_normalization_26/batchnorm/ReadVariableOp/layer_normalization_26/batchnorm/ReadVariableOp2j
3layer_normalization_26/batchnorm/mul/ReadVariableOp3layer_normalization_26/batchnorm/mul/ReadVariableOp2b
/layer_normalization_27/batchnorm/ReadVariableOp/layer_normalization_27/batchnorm/ReadVariableOp2j
3layer_normalization_27/batchnorm/mul/ReadVariableOp3layer_normalization_27/batchnorm/mul/ReadVariableOp2p
6multi_head_attention_8/dense_96/BiasAdd/ReadVariableOp6multi_head_attention_8/dense_96/BiasAdd/ReadVariableOp2t
8multi_head_attention_8/dense_96/Tensordot/ReadVariableOp8multi_head_attention_8/dense_96/Tensordot/ReadVariableOp2p
6multi_head_attention_8/dense_97/BiasAdd/ReadVariableOp6multi_head_attention_8/dense_97/BiasAdd/ReadVariableOp2t
8multi_head_attention_8/dense_97/Tensordot/ReadVariableOp8multi_head_attention_8/dense_97/Tensordot/ReadVariableOp2p
6multi_head_attention_8/dense_98/BiasAdd/ReadVariableOp6multi_head_attention_8/dense_98/BiasAdd/ReadVariableOp2t
8multi_head_attention_8/dense_98/Tensordot/ReadVariableOp8multi_head_attention_8/dense_98/Tensordot/ReadVariableOp2p
6multi_head_attention_8/dense_99/BiasAdd/ReadVariableOp6multi_head_attention_8/dense_99/BiasAdd/ReadVariableOp2t
8multi_head_attention_8/dense_99/Tensordot/ReadVariableOp8multi_head_attention_8/dense_99/Tensordot/ReadVariableOp2^
-sequential_8/dense_100/BiasAdd/ReadVariableOp-sequential_8/dense_100/BiasAdd/ReadVariableOp2b
/sequential_8/dense_100/Tensordot/ReadVariableOp/sequential_8/dense_100/Tensordot/ReadVariableOp2^
-sequential_8/dense_101/BiasAdd/ReadVariableOp-sequential_8/dense_101/BiasAdd/ReadVariableOp2b
/sequential_8/dense_101/Tensordot/ReadVariableOp/sequential_8/dense_101/Tensordot/ReadVariableOp:S O
+
_output_shapes
:?????????d@
 
_user_specified_nameinputs
?
?
)__inference_dense_103_layer_call_fn_93633

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_103_layer_call_and_return_conditional_losses_913922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
 __inference__wrapped_model_90811
input_11\
Jmodel_8_token_and_position_embedding_9_embedding_21_embedding_lookup_90540:d@]
Jmodel_8_token_and_position_embedding_9_embedding_20_embedding_lookup_90546:	?@@o
]model_8_transformer_block_8_multi_head_attention_8_dense_96_tensordot_readvariableop_resource:@@i
[model_8_transformer_block_8_multi_head_attention_8_dense_96_biasadd_readvariableop_resource:@o
]model_8_transformer_block_8_multi_head_attention_8_dense_97_tensordot_readvariableop_resource:@@i
[model_8_transformer_block_8_multi_head_attention_8_dense_97_biasadd_readvariableop_resource:@o
]model_8_transformer_block_8_multi_head_attention_8_dense_98_tensordot_readvariableop_resource:@@i
[model_8_transformer_block_8_multi_head_attention_8_dense_98_biasadd_readvariableop_resource:@o
]model_8_transformer_block_8_multi_head_attention_8_dense_99_tensordot_readvariableop_resource:@@i
[model_8_transformer_block_8_multi_head_attention_8_dense_99_biasadd_readvariableop_resource:@f
Xmodel_8_transformer_block_8_layer_normalization_26_batchnorm_mul_readvariableop_resource:@b
Tmodel_8_transformer_block_8_layer_normalization_26_batchnorm_readvariableop_resource:@f
Tmodel_8_transformer_block_8_sequential_8_dense_100_tensordot_readvariableop_resource:@ `
Rmodel_8_transformer_block_8_sequential_8_dense_100_biasadd_readvariableop_resource: f
Tmodel_8_transformer_block_8_sequential_8_dense_101_tensordot_readvariableop_resource: @`
Rmodel_8_transformer_block_8_sequential_8_dense_101_biasadd_readvariableop_resource:@f
Xmodel_8_transformer_block_8_layer_normalization_27_batchnorm_mul_readvariableop_resource:@b
Tmodel_8_transformer_block_8_layer_normalization_27_batchnorm_readvariableop_resource:@B
0model_8_dense_102_matmul_readvariableop_resource:@?
1model_8_dense_102_biasadd_readvariableop_resource:B
0model_8_dense_103_matmul_readvariableop_resource:?
1model_8_dense_103_biasadd_readvariableop_resource:
identity??(model_8/dense_102/BiasAdd/ReadVariableOp?'model_8/dense_102/MatMul/ReadVariableOp?(model_8/dense_103/BiasAdd/ReadVariableOp?'model_8/dense_103/MatMul/ReadVariableOp?Dmodel_8/token_and_position_embedding_9/embedding_20/embedding_lookup?Dmodel_8/token_and_position_embedding_9/embedding_21/embedding_lookup?Kmodel_8/transformer_block_8/layer_normalization_26/batchnorm/ReadVariableOp?Omodel_8/transformer_block_8/layer_normalization_26/batchnorm/mul/ReadVariableOp?Kmodel_8/transformer_block_8/layer_normalization_27/batchnorm/ReadVariableOp?Omodel_8/transformer_block_8/layer_normalization_27/batchnorm/mul/ReadVariableOp?Rmodel_8/transformer_block_8/multi_head_attention_8/dense_96/BiasAdd/ReadVariableOp?Tmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/ReadVariableOp?Rmodel_8/transformer_block_8/multi_head_attention_8/dense_97/BiasAdd/ReadVariableOp?Tmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/ReadVariableOp?Rmodel_8/transformer_block_8/multi_head_attention_8/dense_98/BiasAdd/ReadVariableOp?Tmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/ReadVariableOp?Rmodel_8/transformer_block_8/multi_head_attention_8/dense_99/BiasAdd/ReadVariableOp?Tmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/ReadVariableOp?Imodel_8/transformer_block_8/sequential_8/dense_100/BiasAdd/ReadVariableOp?Kmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/ReadVariableOp?Imodel_8/transformer_block_8/sequential_8/dense_101/BiasAdd/ReadVariableOp?Kmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/ReadVariableOp?
,model_8/token_and_position_embedding_9/ShapeShapeinput_11*
T0*
_output_shapes
:2.
,model_8/token_and_position_embedding_9/Shape?
:model_8/token_and_position_embedding_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2<
:model_8/token_and_position_embedding_9/strided_slice/stack?
<model_8/token_and_position_embedding_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<model_8/token_and_position_embedding_9/strided_slice/stack_1?
<model_8/token_and_position_embedding_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<model_8/token_and_position_embedding_9/strided_slice/stack_2?
4model_8/token_and_position_embedding_9/strided_sliceStridedSlice5model_8/token_and_position_embedding_9/Shape:output:0Cmodel_8/token_and_position_embedding_9/strided_slice/stack:output:0Emodel_8/token_and_position_embedding_9/strided_slice/stack_1:output:0Emodel_8/token_and_position_embedding_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4model_8/token_and_position_embedding_9/strided_slice?
2model_8/token_and_position_embedding_9/range/startConst*
_output_shapes
: *
dtype0*
value	B : 24
2model_8/token_and_position_embedding_9/range/start?
2model_8/token_and_position_embedding_9/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :24
2model_8/token_and_position_embedding_9/range/delta?
,model_8/token_and_position_embedding_9/rangeRange;model_8/token_and_position_embedding_9/range/start:output:0=model_8/token_and_position_embedding_9/strided_slice:output:0;model_8/token_and_position_embedding_9/range/delta:output:0*#
_output_shapes
:?????????2.
,model_8/token_and_position_embedding_9/range?
Dmodel_8/token_and_position_embedding_9/embedding_21/embedding_lookupResourceGatherJmodel_8_token_and_position_embedding_9_embedding_21_embedding_lookup_905405model_8/token_and_position_embedding_9/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*]
_classS
QOloc:@model_8/token_and_position_embedding_9/embedding_21/embedding_lookup/90540*'
_output_shapes
:?????????@*
dtype02F
Dmodel_8/token_and_position_embedding_9/embedding_21/embedding_lookup?
Mmodel_8/token_and_position_embedding_9/embedding_21/embedding_lookup/IdentityIdentityMmodel_8/token_and_position_embedding_9/embedding_21/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*]
_classS
QOloc:@model_8/token_and_position_embedding_9/embedding_21/embedding_lookup/90540*'
_output_shapes
:?????????@2O
Mmodel_8/token_and_position_embedding_9/embedding_21/embedding_lookup/Identity?
Omodel_8/token_and_position_embedding_9/embedding_21/embedding_lookup/Identity_1IdentityVmodel_8/token_and_position_embedding_9/embedding_21/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????@2Q
Omodel_8/token_and_position_embedding_9/embedding_21/embedding_lookup/Identity_1?
8model_8/token_and_position_embedding_9/embedding_20/CastCastinput_11*

DstT0*

SrcT0*'
_output_shapes
:?????????d2:
8model_8/token_and_position_embedding_9/embedding_20/Cast?
Dmodel_8/token_and_position_embedding_9/embedding_20/embedding_lookupResourceGatherJmodel_8_token_and_position_embedding_9_embedding_20_embedding_lookup_90546<model_8/token_and_position_embedding_9/embedding_20/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*]
_classS
QOloc:@model_8/token_and_position_embedding_9/embedding_20/embedding_lookup/90546*+
_output_shapes
:?????????d@*
dtype02F
Dmodel_8/token_and_position_embedding_9/embedding_20/embedding_lookup?
Mmodel_8/token_and_position_embedding_9/embedding_20/embedding_lookup/IdentityIdentityMmodel_8/token_and_position_embedding_9/embedding_20/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*]
_classS
QOloc:@model_8/token_and_position_embedding_9/embedding_20/embedding_lookup/90546*+
_output_shapes
:?????????d@2O
Mmodel_8/token_and_position_embedding_9/embedding_20/embedding_lookup/Identity?
Omodel_8/token_and_position_embedding_9/embedding_20/embedding_lookup/Identity_1IdentityVmodel_8/token_and_position_embedding_9/embedding_20/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????d@2Q
Omodel_8/token_and_position_embedding_9/embedding_20/embedding_lookup/Identity_1?
*model_8/token_and_position_embedding_9/addAddV2Xmodel_8/token_and_position_embedding_9/embedding_20/embedding_lookup/Identity_1:output:0Xmodel_8/token_and_position_embedding_9/embedding_21/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:?????????d@2,
*model_8/token_and_position_embedding_9/add?
8model_8/transformer_block_8/multi_head_attention_8/ShapeShape.model_8/token_and_position_embedding_9/add:z:0*
T0*
_output_shapes
:2:
8model_8/transformer_block_8/multi_head_attention_8/Shape?
Fmodel_8/transformer_block_8/multi_head_attention_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2H
Fmodel_8/transformer_block_8/multi_head_attention_8/strided_slice/stack?
Hmodel_8/transformer_block_8/multi_head_attention_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2J
Hmodel_8/transformer_block_8/multi_head_attention_8/strided_slice/stack_1?
Hmodel_8/transformer_block_8/multi_head_attention_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2J
Hmodel_8/transformer_block_8/multi_head_attention_8/strided_slice/stack_2?
@model_8/transformer_block_8/multi_head_attention_8/strided_sliceStridedSliceAmodel_8/transformer_block_8/multi_head_attention_8/Shape:output:0Omodel_8/transformer_block_8/multi_head_attention_8/strided_slice/stack:output:0Qmodel_8/transformer_block_8/multi_head_attention_8/strided_slice/stack_1:output:0Qmodel_8/transformer_block_8/multi_head_attention_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2B
@model_8/transformer_block_8/multi_head_attention_8/strided_slice?
Tmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/ReadVariableOpReadVariableOp]model_8_transformer_block_8_multi_head_attention_8_dense_96_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02V
Tmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/ReadVariableOp?
Jmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2L
Jmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/axes?
Jmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2L
Jmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/free?
Kmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/ShapeShape.model_8/token_and_position_embedding_9/add:z:0*
T0*
_output_shapes
:2M
Kmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/Shape?
Smodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2U
Smodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/GatherV2/axis?
Nmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/GatherV2GatherV2Tmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/Shape:output:0Smodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/free:output:0\model_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2P
Nmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/GatherV2?
Umodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2W
Umodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/GatherV2_1/axis?
Pmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/GatherV2_1GatherV2Tmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/Shape:output:0Smodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/axes:output:0^model_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2R
Pmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/GatherV2_1?
Kmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2M
Kmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/Const?
Jmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/ProdProdWmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/GatherV2:output:0Tmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/Const:output:0*
T0*
_output_shapes
: 2L
Jmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/Prod?
Mmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2O
Mmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/Const_1?
Lmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/Prod_1ProdYmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/GatherV2_1:output:0Vmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2N
Lmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/Prod_1?
Qmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2S
Qmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/concat/axis?
Lmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/concatConcatV2Smodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/free:output:0Smodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/axes:output:0Zmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2N
Lmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/concat?
Kmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/stackPackSmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/Prod:output:0Umodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2M
Kmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/stack?
Omodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/transpose	Transpose.model_8/token_and_position_embedding_9/add:z:0Umodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????d@2Q
Omodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/transpose?
Mmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/ReshapeReshapeSmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/transpose:y:0Tmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2O
Mmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/Reshape?
Lmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/MatMulMatMulVmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/Reshape:output:0\model_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2N
Lmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/MatMul?
Mmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2O
Mmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/Const_2?
Smodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2U
Smodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/concat_1/axis?
Nmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/concat_1ConcatV2Wmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/GatherV2:output:0Vmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/Const_2:output:0\model_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2P
Nmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/concat_1?
Emodel_8/transformer_block_8/multi_head_attention_8/dense_96/TensordotReshapeVmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/MatMul:product:0Wmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????d@2G
Emodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot?
Rmodel_8/transformer_block_8/multi_head_attention_8/dense_96/BiasAdd/ReadVariableOpReadVariableOp[model_8_transformer_block_8_multi_head_attention_8_dense_96_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02T
Rmodel_8/transformer_block_8/multi_head_attention_8/dense_96/BiasAdd/ReadVariableOp?
Cmodel_8/transformer_block_8/multi_head_attention_8/dense_96/BiasAddBiasAddNmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot:output:0Zmodel_8/transformer_block_8/multi_head_attention_8/dense_96/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@2E
Cmodel_8/transformer_block_8/multi_head_attention_8/dense_96/BiasAdd?
Tmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/ReadVariableOpReadVariableOp]model_8_transformer_block_8_multi_head_attention_8_dense_97_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02V
Tmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/ReadVariableOp?
Jmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2L
Jmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/axes?
Jmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2L
Jmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/free?
Kmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/ShapeShape.model_8/token_and_position_embedding_9/add:z:0*
T0*
_output_shapes
:2M
Kmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/Shape?
Smodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2U
Smodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/GatherV2/axis?
Nmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/GatherV2GatherV2Tmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/Shape:output:0Smodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/free:output:0\model_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2P
Nmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/GatherV2?
Umodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2W
Umodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/GatherV2_1/axis?
Pmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/GatherV2_1GatherV2Tmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/Shape:output:0Smodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/axes:output:0^model_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2R
Pmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/GatherV2_1?
Kmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2M
Kmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/Const?
Jmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/ProdProdWmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/GatherV2:output:0Tmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/Const:output:0*
T0*
_output_shapes
: 2L
Jmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/Prod?
Mmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2O
Mmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/Const_1?
Lmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/Prod_1ProdYmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/GatherV2_1:output:0Vmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2N
Lmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/Prod_1?
Qmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2S
Qmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/concat/axis?
Lmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/concatConcatV2Smodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/free:output:0Smodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/axes:output:0Zmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2N
Lmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/concat?
Kmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/stackPackSmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/Prod:output:0Umodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2M
Kmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/stack?
Omodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/transpose	Transpose.model_8/token_and_position_embedding_9/add:z:0Umodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????d@2Q
Omodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/transpose?
Mmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/ReshapeReshapeSmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/transpose:y:0Tmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2O
Mmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/Reshape?
Lmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/MatMulMatMulVmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/Reshape:output:0\model_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2N
Lmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/MatMul?
Mmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2O
Mmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/Const_2?
Smodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2U
Smodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/concat_1/axis?
Nmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/concat_1ConcatV2Wmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/GatherV2:output:0Vmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/Const_2:output:0\model_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2P
Nmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/concat_1?
Emodel_8/transformer_block_8/multi_head_attention_8/dense_97/TensordotReshapeVmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/MatMul:product:0Wmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????d@2G
Emodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot?
Rmodel_8/transformer_block_8/multi_head_attention_8/dense_97/BiasAdd/ReadVariableOpReadVariableOp[model_8_transformer_block_8_multi_head_attention_8_dense_97_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02T
Rmodel_8/transformer_block_8/multi_head_attention_8/dense_97/BiasAdd/ReadVariableOp?
Cmodel_8/transformer_block_8/multi_head_attention_8/dense_97/BiasAddBiasAddNmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot:output:0Zmodel_8/transformer_block_8/multi_head_attention_8/dense_97/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@2E
Cmodel_8/transformer_block_8/multi_head_attention_8/dense_97/BiasAdd?
Tmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/ReadVariableOpReadVariableOp]model_8_transformer_block_8_multi_head_attention_8_dense_98_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02V
Tmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/ReadVariableOp?
Jmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2L
Jmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/axes?
Jmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2L
Jmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/free?
Kmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/ShapeShape.model_8/token_and_position_embedding_9/add:z:0*
T0*
_output_shapes
:2M
Kmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/Shape?
Smodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2U
Smodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/GatherV2/axis?
Nmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/GatherV2GatherV2Tmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/Shape:output:0Smodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/free:output:0\model_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2P
Nmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/GatherV2?
Umodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2W
Umodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/GatherV2_1/axis?
Pmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/GatherV2_1GatherV2Tmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/Shape:output:0Smodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/axes:output:0^model_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2R
Pmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/GatherV2_1?
Kmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2M
Kmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/Const?
Jmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/ProdProdWmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/GatherV2:output:0Tmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/Const:output:0*
T0*
_output_shapes
: 2L
Jmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/Prod?
Mmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2O
Mmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/Const_1?
Lmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/Prod_1ProdYmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/GatherV2_1:output:0Vmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2N
Lmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/Prod_1?
Qmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2S
Qmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/concat/axis?
Lmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/concatConcatV2Smodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/free:output:0Smodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/axes:output:0Zmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2N
Lmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/concat?
Kmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/stackPackSmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/Prod:output:0Umodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2M
Kmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/stack?
Omodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/transpose	Transpose.model_8/token_and_position_embedding_9/add:z:0Umodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????d@2Q
Omodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/transpose?
Mmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/ReshapeReshapeSmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/transpose:y:0Tmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2O
Mmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/Reshape?
Lmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/MatMulMatMulVmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/Reshape:output:0\model_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2N
Lmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/MatMul?
Mmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2O
Mmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/Const_2?
Smodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2U
Smodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/concat_1/axis?
Nmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/concat_1ConcatV2Wmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/GatherV2:output:0Vmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/Const_2:output:0\model_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2P
Nmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/concat_1?
Emodel_8/transformer_block_8/multi_head_attention_8/dense_98/TensordotReshapeVmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/MatMul:product:0Wmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????d@2G
Emodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot?
Rmodel_8/transformer_block_8/multi_head_attention_8/dense_98/BiasAdd/ReadVariableOpReadVariableOp[model_8_transformer_block_8_multi_head_attention_8_dense_98_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02T
Rmodel_8/transformer_block_8/multi_head_attention_8/dense_98/BiasAdd/ReadVariableOp?
Cmodel_8/transformer_block_8/multi_head_attention_8/dense_98/BiasAddBiasAddNmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot:output:0Zmodel_8/transformer_block_8/multi_head_attention_8/dense_98/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@2E
Cmodel_8/transformer_block_8/multi_head_attention_8/dense_98/BiasAdd?
Bmodel_8/transformer_block_8/multi_head_attention_8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2D
Bmodel_8/transformer_block_8/multi_head_attention_8/Reshape/shape/1?
Bmodel_8/transformer_block_8/multi_head_attention_8/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2D
Bmodel_8/transformer_block_8/multi_head_attention_8/Reshape/shape/2?
Bmodel_8/transformer_block_8/multi_head_attention_8/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2D
Bmodel_8/transformer_block_8/multi_head_attention_8/Reshape/shape/3?
@model_8/transformer_block_8/multi_head_attention_8/Reshape/shapePackImodel_8/transformer_block_8/multi_head_attention_8/strided_slice:output:0Kmodel_8/transformer_block_8/multi_head_attention_8/Reshape/shape/1:output:0Kmodel_8/transformer_block_8/multi_head_attention_8/Reshape/shape/2:output:0Kmodel_8/transformer_block_8/multi_head_attention_8/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2B
@model_8/transformer_block_8/multi_head_attention_8/Reshape/shape?
:model_8/transformer_block_8/multi_head_attention_8/ReshapeReshapeLmodel_8/transformer_block_8/multi_head_attention_8/dense_96/BiasAdd:output:0Imodel_8/transformer_block_8/multi_head_attention_8/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2<
:model_8/transformer_block_8/multi_head_attention_8/Reshape?
Amodel_8/transformer_block_8/multi_head_attention_8/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2C
Amodel_8/transformer_block_8/multi_head_attention_8/transpose/perm?
<model_8/transformer_block_8/multi_head_attention_8/transpose	TransposeCmodel_8/transformer_block_8/multi_head_attention_8/Reshape:output:0Jmodel_8/transformer_block_8/multi_head_attention_8/transpose/perm:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2>
<model_8/transformer_block_8/multi_head_attention_8/transpose?
Dmodel_8/transformer_block_8/multi_head_attention_8/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2F
Dmodel_8/transformer_block_8/multi_head_attention_8/Reshape_1/shape/1?
Dmodel_8/transformer_block_8/multi_head_attention_8/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2F
Dmodel_8/transformer_block_8/multi_head_attention_8/Reshape_1/shape/2?
Dmodel_8/transformer_block_8/multi_head_attention_8/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2F
Dmodel_8/transformer_block_8/multi_head_attention_8/Reshape_1/shape/3?
Bmodel_8/transformer_block_8/multi_head_attention_8/Reshape_1/shapePackImodel_8/transformer_block_8/multi_head_attention_8/strided_slice:output:0Mmodel_8/transformer_block_8/multi_head_attention_8/Reshape_1/shape/1:output:0Mmodel_8/transformer_block_8/multi_head_attention_8/Reshape_1/shape/2:output:0Mmodel_8/transformer_block_8/multi_head_attention_8/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2D
Bmodel_8/transformer_block_8/multi_head_attention_8/Reshape_1/shape?
<model_8/transformer_block_8/multi_head_attention_8/Reshape_1ReshapeLmodel_8/transformer_block_8/multi_head_attention_8/dense_97/BiasAdd:output:0Kmodel_8/transformer_block_8/multi_head_attention_8/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2>
<model_8/transformer_block_8/multi_head_attention_8/Reshape_1?
Cmodel_8/transformer_block_8/multi_head_attention_8/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2E
Cmodel_8/transformer_block_8/multi_head_attention_8/transpose_1/perm?
>model_8/transformer_block_8/multi_head_attention_8/transpose_1	TransposeEmodel_8/transformer_block_8/multi_head_attention_8/Reshape_1:output:0Lmodel_8/transformer_block_8/multi_head_attention_8/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2@
>model_8/transformer_block_8/multi_head_attention_8/transpose_1?
Dmodel_8/transformer_block_8/multi_head_attention_8/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2F
Dmodel_8/transformer_block_8/multi_head_attention_8/Reshape_2/shape/1?
Dmodel_8/transformer_block_8/multi_head_attention_8/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2F
Dmodel_8/transformer_block_8/multi_head_attention_8/Reshape_2/shape/2?
Dmodel_8/transformer_block_8/multi_head_attention_8/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2F
Dmodel_8/transformer_block_8/multi_head_attention_8/Reshape_2/shape/3?
Bmodel_8/transformer_block_8/multi_head_attention_8/Reshape_2/shapePackImodel_8/transformer_block_8/multi_head_attention_8/strided_slice:output:0Mmodel_8/transformer_block_8/multi_head_attention_8/Reshape_2/shape/1:output:0Mmodel_8/transformer_block_8/multi_head_attention_8/Reshape_2/shape/2:output:0Mmodel_8/transformer_block_8/multi_head_attention_8/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:2D
Bmodel_8/transformer_block_8/multi_head_attention_8/Reshape_2/shape?
<model_8/transformer_block_8/multi_head_attention_8/Reshape_2ReshapeLmodel_8/transformer_block_8/multi_head_attention_8/dense_98/BiasAdd:output:0Kmodel_8/transformer_block_8/multi_head_attention_8/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2>
<model_8/transformer_block_8/multi_head_attention_8/Reshape_2?
Cmodel_8/transformer_block_8/multi_head_attention_8/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2E
Cmodel_8/transformer_block_8/multi_head_attention_8/transpose_2/perm?
>model_8/transformer_block_8/multi_head_attention_8/transpose_2	TransposeEmodel_8/transformer_block_8/multi_head_attention_8/Reshape_2:output:0Lmodel_8/transformer_block_8/multi_head_attention_8/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2@
>model_8/transformer_block_8/multi_head_attention_8/transpose_2?
9model_8/transformer_block_8/multi_head_attention_8/MatMulBatchMatMulV2@model_8/transformer_block_8/multi_head_attention_8/transpose:y:0Bmodel_8/transformer_block_8/multi_head_attention_8/transpose_1:y:0*
T0*A
_output_shapes/
-:+???????????????????????????*
adj_y(2;
9model_8/transformer_block_8/multi_head_attention_8/MatMul?
:model_8/transformer_block_8/multi_head_attention_8/Shape_1ShapeBmodel_8/transformer_block_8/multi_head_attention_8/transpose_1:y:0*
T0*
_output_shapes
:2<
:model_8/transformer_block_8/multi_head_attention_8/Shape_1?
Hmodel_8/transformer_block_8/multi_head_attention_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2J
Hmodel_8/transformer_block_8/multi_head_attention_8/strided_slice_1/stack?
Jmodel_8/transformer_block_8/multi_head_attention_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2L
Jmodel_8/transformer_block_8/multi_head_attention_8/strided_slice_1/stack_1?
Jmodel_8/transformer_block_8/multi_head_attention_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2L
Jmodel_8/transformer_block_8/multi_head_attention_8/strided_slice_1/stack_2?
Bmodel_8/transformer_block_8/multi_head_attention_8/strided_slice_1StridedSliceCmodel_8/transformer_block_8/multi_head_attention_8/Shape_1:output:0Qmodel_8/transformer_block_8/multi_head_attention_8/strided_slice_1/stack:output:0Smodel_8/transformer_block_8/multi_head_attention_8/strided_slice_1/stack_1:output:0Smodel_8/transformer_block_8/multi_head_attention_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2D
Bmodel_8/transformer_block_8/multi_head_attention_8/strided_slice_1?
7model_8/transformer_block_8/multi_head_attention_8/CastCastKmodel_8/transformer_block_8/multi_head_attention_8/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 29
7model_8/transformer_block_8/multi_head_attention_8/Cast?
7model_8/transformer_block_8/multi_head_attention_8/SqrtSqrt;model_8/transformer_block_8/multi_head_attention_8/Cast:y:0*
T0*
_output_shapes
: 29
7model_8/transformer_block_8/multi_head_attention_8/Sqrt?
:model_8/transformer_block_8/multi_head_attention_8/truedivRealDivBmodel_8/transformer_block_8/multi_head_attention_8/MatMul:output:0;model_8/transformer_block_8/multi_head_attention_8/Sqrt:y:0*
T0*A
_output_shapes/
-:+???????????????????????????2<
:model_8/transformer_block_8/multi_head_attention_8/truediv?
:model_8/transformer_block_8/multi_head_attention_8/SoftmaxSoftmax>model_8/transformer_block_8/multi_head_attention_8/truediv:z:0*
T0*A
_output_shapes/
-:+???????????????????????????2<
:model_8/transformer_block_8/multi_head_attention_8/Softmax?
;model_8/transformer_block_8/multi_head_attention_8/MatMul_1BatchMatMulV2Dmodel_8/transformer_block_8/multi_head_attention_8/Softmax:softmax:0Bmodel_8/transformer_block_8/multi_head_attention_8/transpose_2:y:0*
T0*8
_output_shapes&
$:"?????????????????? 2=
;model_8/transformer_block_8/multi_head_attention_8/MatMul_1?
Cmodel_8/transformer_block_8/multi_head_attention_8/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             2E
Cmodel_8/transformer_block_8/multi_head_attention_8/transpose_3/perm?
>model_8/transformer_block_8/multi_head_attention_8/transpose_3	TransposeDmodel_8/transformer_block_8/multi_head_attention_8/MatMul_1:output:0Lmodel_8/transformer_block_8/multi_head_attention_8/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"?????????????????? 2@
>model_8/transformer_block_8/multi_head_attention_8/transpose_3?
Dmodel_8/transformer_block_8/multi_head_attention_8/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2F
Dmodel_8/transformer_block_8/multi_head_attention_8/Reshape_3/shape/1?
Dmodel_8/transformer_block_8/multi_head_attention_8/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@2F
Dmodel_8/transformer_block_8/multi_head_attention_8/Reshape_3/shape/2?
Bmodel_8/transformer_block_8/multi_head_attention_8/Reshape_3/shapePackImodel_8/transformer_block_8/multi_head_attention_8/strided_slice:output:0Mmodel_8/transformer_block_8/multi_head_attention_8/Reshape_3/shape/1:output:0Mmodel_8/transformer_block_8/multi_head_attention_8/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2D
Bmodel_8/transformer_block_8/multi_head_attention_8/Reshape_3/shape?
<model_8/transformer_block_8/multi_head_attention_8/Reshape_3ReshapeBmodel_8/transformer_block_8/multi_head_attention_8/transpose_3:y:0Kmodel_8/transformer_block_8/multi_head_attention_8/Reshape_3/shape:output:0*
T0*4
_output_shapes"
 :??????????????????@2>
<model_8/transformer_block_8/multi_head_attention_8/Reshape_3?
Tmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/ReadVariableOpReadVariableOp]model_8_transformer_block_8_multi_head_attention_8_dense_99_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02V
Tmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/ReadVariableOp?
Jmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2L
Jmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/axes?
Jmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2L
Jmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/free?
Kmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/ShapeShapeEmodel_8/transformer_block_8/multi_head_attention_8/Reshape_3:output:0*
T0*
_output_shapes
:2M
Kmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/Shape?
Smodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2U
Smodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/GatherV2/axis?
Nmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/GatherV2GatherV2Tmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/Shape:output:0Smodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/free:output:0\model_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2P
Nmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/GatherV2?
Umodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2W
Umodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/GatherV2_1/axis?
Pmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/GatherV2_1GatherV2Tmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/Shape:output:0Smodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/axes:output:0^model_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2R
Pmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/GatherV2_1?
Kmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2M
Kmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/Const?
Jmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/ProdProdWmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/GatherV2:output:0Tmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/Const:output:0*
T0*
_output_shapes
: 2L
Jmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/Prod?
Mmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2O
Mmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/Const_1?
Lmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/Prod_1ProdYmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/GatherV2_1:output:0Vmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2N
Lmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/Prod_1?
Qmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2S
Qmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/concat/axis?
Lmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/concatConcatV2Smodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/free:output:0Smodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/axes:output:0Zmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2N
Lmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/concat?
Kmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/stackPackSmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/Prod:output:0Umodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2M
Kmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/stack?
Omodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/transpose	TransposeEmodel_8/transformer_block_8/multi_head_attention_8/Reshape_3:output:0Umodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :??????????????????@2Q
Omodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/transpose?
Mmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/ReshapeReshapeSmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/transpose:y:0Tmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2O
Mmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/Reshape?
Lmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/MatMulMatMulVmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/Reshape:output:0\model_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2N
Lmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/MatMul?
Mmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2O
Mmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/Const_2?
Smodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2U
Smodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/concat_1/axis?
Nmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/concat_1ConcatV2Wmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/GatherV2:output:0Vmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/Const_2:output:0\model_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2P
Nmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/concat_1?
Emodel_8/transformer_block_8/multi_head_attention_8/dense_99/TensordotReshapeVmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/MatMul:product:0Wmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :??????????????????@2G
Emodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot?
Rmodel_8/transformer_block_8/multi_head_attention_8/dense_99/BiasAdd/ReadVariableOpReadVariableOp[model_8_transformer_block_8_multi_head_attention_8_dense_99_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02T
Rmodel_8/transformer_block_8/multi_head_attention_8/dense_99/BiasAdd/ReadVariableOp?
Cmodel_8/transformer_block_8/multi_head_attention_8/dense_99/BiasAddBiasAddNmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot:output:0Zmodel_8/transformer_block_8/multi_head_attention_8/dense_99/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????@2E
Cmodel_8/transformer_block_8/multi_head_attention_8/dense_99/BiasAdd?
/model_8/transformer_block_8/dropout_42/IdentityIdentityLmodel_8/transformer_block_8/multi_head_attention_8/dense_99/BiasAdd:output:0*
T0*4
_output_shapes"
 :??????????????????@21
/model_8/transformer_block_8/dropout_42/Identity?
model_8/transformer_block_8/addAddV2.model_8/token_and_position_embedding_9/add:z:08model_8/transformer_block_8/dropout_42/Identity:output:0*
T0*+
_output_shapes
:?????????d@2!
model_8/transformer_block_8/add?
Qmodel_8/transformer_block_8/layer_normalization_26/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2S
Qmodel_8/transformer_block_8/layer_normalization_26/moments/mean/reduction_indices?
?model_8/transformer_block_8/layer_normalization_26/moments/meanMean#model_8/transformer_block_8/add:z:0Zmodel_8/transformer_block_8/layer_normalization_26/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????d*
	keep_dims(2A
?model_8/transformer_block_8/layer_normalization_26/moments/mean?
Gmodel_8/transformer_block_8/layer_normalization_26/moments/StopGradientStopGradientHmodel_8/transformer_block_8/layer_normalization_26/moments/mean:output:0*
T0*+
_output_shapes
:?????????d2I
Gmodel_8/transformer_block_8/layer_normalization_26/moments/StopGradient?
Lmodel_8/transformer_block_8/layer_normalization_26/moments/SquaredDifferenceSquaredDifference#model_8/transformer_block_8/add:z:0Pmodel_8/transformer_block_8/layer_normalization_26/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????d@2N
Lmodel_8/transformer_block_8/layer_normalization_26/moments/SquaredDifference?
Umodel_8/transformer_block_8/layer_normalization_26/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2W
Umodel_8/transformer_block_8/layer_normalization_26/moments/variance/reduction_indices?
Cmodel_8/transformer_block_8/layer_normalization_26/moments/varianceMeanPmodel_8/transformer_block_8/layer_normalization_26/moments/SquaredDifference:z:0^model_8/transformer_block_8/layer_normalization_26/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????d*
	keep_dims(2E
Cmodel_8/transformer_block_8/layer_normalization_26/moments/variance?
Bmodel_8/transformer_block_8/layer_normalization_26/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52D
Bmodel_8/transformer_block_8/layer_normalization_26/batchnorm/add/y?
@model_8/transformer_block_8/layer_normalization_26/batchnorm/addAddV2Lmodel_8/transformer_block_8/layer_normalization_26/moments/variance:output:0Kmodel_8/transformer_block_8/layer_normalization_26/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????d2B
@model_8/transformer_block_8/layer_normalization_26/batchnorm/add?
Bmodel_8/transformer_block_8/layer_normalization_26/batchnorm/RsqrtRsqrtDmodel_8/transformer_block_8/layer_normalization_26/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????d2D
Bmodel_8/transformer_block_8/layer_normalization_26/batchnorm/Rsqrt?
Omodel_8/transformer_block_8/layer_normalization_26/batchnorm/mul/ReadVariableOpReadVariableOpXmodel_8_transformer_block_8_layer_normalization_26_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02Q
Omodel_8/transformer_block_8/layer_normalization_26/batchnorm/mul/ReadVariableOp?
@model_8/transformer_block_8/layer_normalization_26/batchnorm/mulMulFmodel_8/transformer_block_8/layer_normalization_26/batchnorm/Rsqrt:y:0Wmodel_8/transformer_block_8/layer_normalization_26/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@2B
@model_8/transformer_block_8/layer_normalization_26/batchnorm/mul?
Bmodel_8/transformer_block_8/layer_normalization_26/batchnorm/mul_1Mul#model_8/transformer_block_8/add:z:0Dmodel_8/transformer_block_8/layer_normalization_26/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????d@2D
Bmodel_8/transformer_block_8/layer_normalization_26/batchnorm/mul_1?
Bmodel_8/transformer_block_8/layer_normalization_26/batchnorm/mul_2MulHmodel_8/transformer_block_8/layer_normalization_26/moments/mean:output:0Dmodel_8/transformer_block_8/layer_normalization_26/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????d@2D
Bmodel_8/transformer_block_8/layer_normalization_26/batchnorm/mul_2?
Kmodel_8/transformer_block_8/layer_normalization_26/batchnorm/ReadVariableOpReadVariableOpTmodel_8_transformer_block_8_layer_normalization_26_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02M
Kmodel_8/transformer_block_8/layer_normalization_26/batchnorm/ReadVariableOp?
@model_8/transformer_block_8/layer_normalization_26/batchnorm/subSubSmodel_8/transformer_block_8/layer_normalization_26/batchnorm/ReadVariableOp:value:0Fmodel_8/transformer_block_8/layer_normalization_26/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????d@2B
@model_8/transformer_block_8/layer_normalization_26/batchnorm/sub?
Bmodel_8/transformer_block_8/layer_normalization_26/batchnorm/add_1AddV2Fmodel_8/transformer_block_8/layer_normalization_26/batchnorm/mul_1:z:0Dmodel_8/transformer_block_8/layer_normalization_26/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????d@2D
Bmodel_8/transformer_block_8/layer_normalization_26/batchnorm/add_1?
Kmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/ReadVariableOpReadVariableOpTmodel_8_transformer_block_8_sequential_8_dense_100_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02M
Kmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/ReadVariableOp?
Amodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2C
Amodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/axes?
Amodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2C
Amodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/free?
Bmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/ShapeShapeFmodel_8/transformer_block_8/layer_normalization_26/batchnorm/add_1:z:0*
T0*
_output_shapes
:2D
Bmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/Shape?
Jmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2L
Jmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/GatherV2/axis?
Emodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/GatherV2GatherV2Kmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/Shape:output:0Jmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/free:output:0Smodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2G
Emodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/GatherV2?
Lmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2N
Lmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/GatherV2_1/axis?
Gmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/GatherV2_1GatherV2Kmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/Shape:output:0Jmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/axes:output:0Umodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2I
Gmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/GatherV2_1?
Bmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/Const?
Amodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/ProdProdNmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/GatherV2:output:0Kmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/Const:output:0*
T0*
_output_shapes
: 2C
Amodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/Prod?
Dmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2F
Dmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/Const_1?
Cmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/Prod_1ProdPmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/GatherV2_1:output:0Mmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2E
Cmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/Prod_1?
Hmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/concat/axis?
Cmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/concatConcatV2Jmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/free:output:0Jmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/axes:output:0Qmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2E
Cmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/concat?
Bmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/stackPackJmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/Prod:output:0Lmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2D
Bmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/stack?
Fmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/transpose	TransposeFmodel_8/transformer_block_8/layer_normalization_26/batchnorm/add_1:z:0Lmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????d@2H
Fmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/transpose?
Dmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/ReshapeReshapeJmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/transpose:y:0Kmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2F
Dmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/Reshape?
Cmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/MatMulMatMulMmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/Reshape:output:0Smodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2E
Cmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/MatMul?
Dmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2F
Dmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/Const_2?
Jmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2L
Jmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/concat_1/axis?
Emodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/concat_1ConcatV2Nmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/GatherV2:output:0Mmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/Const_2:output:0Smodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2G
Emodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/concat_1?
<model_8/transformer_block_8/sequential_8/dense_100/TensordotReshapeMmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/MatMul:product:0Nmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????d 2>
<model_8/transformer_block_8/sequential_8/dense_100/Tensordot?
Imodel_8/transformer_block_8/sequential_8/dense_100/BiasAdd/ReadVariableOpReadVariableOpRmodel_8_transformer_block_8_sequential_8_dense_100_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02K
Imodel_8/transformer_block_8/sequential_8/dense_100/BiasAdd/ReadVariableOp?
:model_8/transformer_block_8/sequential_8/dense_100/BiasAddBiasAddEmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot:output:0Qmodel_8/transformer_block_8/sequential_8/dense_100/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d 2<
:model_8/transformer_block_8/sequential_8/dense_100/BiasAdd?
7model_8/transformer_block_8/sequential_8/dense_100/ReluReluCmodel_8/transformer_block_8/sequential_8/dense_100/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d 29
7model_8/transformer_block_8/sequential_8/dense_100/Relu?
Kmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/ReadVariableOpReadVariableOpTmodel_8_transformer_block_8_sequential_8_dense_101_tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02M
Kmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/ReadVariableOp?
Amodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2C
Amodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/axes?
Amodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2C
Amodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/free?
Bmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/ShapeShapeEmodel_8/transformer_block_8/sequential_8/dense_100/Relu:activations:0*
T0*
_output_shapes
:2D
Bmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/Shape?
Jmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2L
Jmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/GatherV2/axis?
Emodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/GatherV2GatherV2Kmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/Shape:output:0Jmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/free:output:0Smodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2G
Emodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/GatherV2?
Lmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2N
Lmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/GatherV2_1/axis?
Gmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/GatherV2_1GatherV2Kmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/Shape:output:0Jmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/axes:output:0Umodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2I
Gmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/GatherV2_1?
Bmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/Const?
Amodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/ProdProdNmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/GatherV2:output:0Kmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/Const:output:0*
T0*
_output_shapes
: 2C
Amodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/Prod?
Dmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2F
Dmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/Const_1?
Cmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/Prod_1ProdPmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/GatherV2_1:output:0Mmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2E
Cmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/Prod_1?
Hmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/concat/axis?
Cmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/concatConcatV2Jmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/free:output:0Jmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/axes:output:0Qmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2E
Cmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/concat?
Bmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/stackPackJmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/Prod:output:0Lmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2D
Bmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/stack?
Fmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/transpose	TransposeEmodel_8/transformer_block_8/sequential_8/dense_100/Relu:activations:0Lmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????d 2H
Fmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/transpose?
Dmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/ReshapeReshapeJmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/transpose:y:0Kmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2F
Dmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/Reshape?
Cmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/MatMulMatMulMmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/Reshape:output:0Smodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2E
Cmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/MatMul?
Dmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2F
Dmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/Const_2?
Jmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2L
Jmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/concat_1/axis?
Emodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/concat_1ConcatV2Nmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/GatherV2:output:0Mmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/Const_2:output:0Smodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2G
Emodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/concat_1?
<model_8/transformer_block_8/sequential_8/dense_101/TensordotReshapeMmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/MatMul:product:0Nmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????d@2>
<model_8/transformer_block_8/sequential_8/dense_101/Tensordot?
Imodel_8/transformer_block_8/sequential_8/dense_101/BiasAdd/ReadVariableOpReadVariableOpRmodel_8_transformer_block_8_sequential_8_dense_101_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02K
Imodel_8/transformer_block_8/sequential_8/dense_101/BiasAdd/ReadVariableOp?
:model_8/transformer_block_8/sequential_8/dense_101/BiasAddBiasAddEmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot:output:0Qmodel_8/transformer_block_8/sequential_8/dense_101/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@2<
:model_8/transformer_block_8/sequential_8/dense_101/BiasAdd?
/model_8/transformer_block_8/dropout_43/IdentityIdentityCmodel_8/transformer_block_8/sequential_8/dense_101/BiasAdd:output:0*
T0*+
_output_shapes
:?????????d@21
/model_8/transformer_block_8/dropout_43/Identity?
!model_8/transformer_block_8/add_1AddV2Fmodel_8/transformer_block_8/layer_normalization_26/batchnorm/add_1:z:08model_8/transformer_block_8/dropout_43/Identity:output:0*
T0*+
_output_shapes
:?????????d@2#
!model_8/transformer_block_8/add_1?
Qmodel_8/transformer_block_8/layer_normalization_27/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2S
Qmodel_8/transformer_block_8/layer_normalization_27/moments/mean/reduction_indices?
?model_8/transformer_block_8/layer_normalization_27/moments/meanMean%model_8/transformer_block_8/add_1:z:0Zmodel_8/transformer_block_8/layer_normalization_27/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????d*
	keep_dims(2A
?model_8/transformer_block_8/layer_normalization_27/moments/mean?
Gmodel_8/transformer_block_8/layer_normalization_27/moments/StopGradientStopGradientHmodel_8/transformer_block_8/layer_normalization_27/moments/mean:output:0*
T0*+
_output_shapes
:?????????d2I
Gmodel_8/transformer_block_8/layer_normalization_27/moments/StopGradient?
Lmodel_8/transformer_block_8/layer_normalization_27/moments/SquaredDifferenceSquaredDifference%model_8/transformer_block_8/add_1:z:0Pmodel_8/transformer_block_8/layer_normalization_27/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????d@2N
Lmodel_8/transformer_block_8/layer_normalization_27/moments/SquaredDifference?
Umodel_8/transformer_block_8/layer_normalization_27/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2W
Umodel_8/transformer_block_8/layer_normalization_27/moments/variance/reduction_indices?
Cmodel_8/transformer_block_8/layer_normalization_27/moments/varianceMeanPmodel_8/transformer_block_8/layer_normalization_27/moments/SquaredDifference:z:0^model_8/transformer_block_8/layer_normalization_27/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:?????????d*
	keep_dims(2E
Cmodel_8/transformer_block_8/layer_normalization_27/moments/variance?
Bmodel_8/transformer_block_8/layer_normalization_27/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52D
Bmodel_8/transformer_block_8/layer_normalization_27/batchnorm/add/y?
@model_8/transformer_block_8/layer_normalization_27/batchnorm/addAddV2Lmodel_8/transformer_block_8/layer_normalization_27/moments/variance:output:0Kmodel_8/transformer_block_8/layer_normalization_27/batchnorm/add/y:output:0*
T0*+
_output_shapes
:?????????d2B
@model_8/transformer_block_8/layer_normalization_27/batchnorm/add?
Bmodel_8/transformer_block_8/layer_normalization_27/batchnorm/RsqrtRsqrtDmodel_8/transformer_block_8/layer_normalization_27/batchnorm/add:z:0*
T0*+
_output_shapes
:?????????d2D
Bmodel_8/transformer_block_8/layer_normalization_27/batchnorm/Rsqrt?
Omodel_8/transformer_block_8/layer_normalization_27/batchnorm/mul/ReadVariableOpReadVariableOpXmodel_8_transformer_block_8_layer_normalization_27_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02Q
Omodel_8/transformer_block_8/layer_normalization_27/batchnorm/mul/ReadVariableOp?
@model_8/transformer_block_8/layer_normalization_27/batchnorm/mulMulFmodel_8/transformer_block_8/layer_normalization_27/batchnorm/Rsqrt:y:0Wmodel_8/transformer_block_8/layer_normalization_27/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@2B
@model_8/transformer_block_8/layer_normalization_27/batchnorm/mul?
Bmodel_8/transformer_block_8/layer_normalization_27/batchnorm/mul_1Mul%model_8/transformer_block_8/add_1:z:0Dmodel_8/transformer_block_8/layer_normalization_27/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????d@2D
Bmodel_8/transformer_block_8/layer_normalization_27/batchnorm/mul_1?
Bmodel_8/transformer_block_8/layer_normalization_27/batchnorm/mul_2MulHmodel_8/transformer_block_8/layer_normalization_27/moments/mean:output:0Dmodel_8/transformer_block_8/layer_normalization_27/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????d@2D
Bmodel_8/transformer_block_8/layer_normalization_27/batchnorm/mul_2?
Kmodel_8/transformer_block_8/layer_normalization_27/batchnorm/ReadVariableOpReadVariableOpTmodel_8_transformer_block_8_layer_normalization_27_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02M
Kmodel_8/transformer_block_8/layer_normalization_27/batchnorm/ReadVariableOp?
@model_8/transformer_block_8/layer_normalization_27/batchnorm/subSubSmodel_8/transformer_block_8/layer_normalization_27/batchnorm/ReadVariableOp:value:0Fmodel_8/transformer_block_8/layer_normalization_27/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:?????????d@2B
@model_8/transformer_block_8/layer_normalization_27/batchnorm/sub?
Bmodel_8/transformer_block_8/layer_normalization_27/batchnorm/add_1AddV2Fmodel_8/transformer_block_8/layer_normalization_27/batchnorm/mul_1:z:0Dmodel_8/transformer_block_8/layer_normalization_27/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????d@2D
Bmodel_8/transformer_block_8/layer_normalization_27/batchnorm/add_1?
9model_8/global_average_pooling1d_8/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2;
9model_8/global_average_pooling1d_8/Mean/reduction_indices?
'model_8/global_average_pooling1d_8/MeanMeanFmodel_8/transformer_block_8/layer_normalization_27/batchnorm/add_1:z:0Bmodel_8/global_average_pooling1d_8/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@2)
'model_8/global_average_pooling1d_8/Mean?
model_8/dropout_44/IdentityIdentity0model_8/global_average_pooling1d_8/Mean:output:0*
T0*'
_output_shapes
:?????????@2
model_8/dropout_44/Identity?
'model_8/dense_102/MatMul/ReadVariableOpReadVariableOp0model_8_dense_102_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02)
'model_8/dense_102/MatMul/ReadVariableOp?
model_8/dense_102/MatMulMatMul$model_8/dropout_44/Identity:output:0/model_8/dense_102/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_8/dense_102/MatMul?
(model_8/dense_102/BiasAdd/ReadVariableOpReadVariableOp1model_8_dense_102_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_8/dense_102/BiasAdd/ReadVariableOp?
model_8/dense_102/BiasAddBiasAdd"model_8/dense_102/MatMul:product:00model_8/dense_102/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_8/dense_102/BiasAdd?
model_8/dense_102/ReluRelu"model_8/dense_102/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_8/dense_102/Relu?
model_8/dropout_45/IdentityIdentity$model_8/dense_102/Relu:activations:0*
T0*'
_output_shapes
:?????????2
model_8/dropout_45/Identity?
'model_8/dense_103/MatMul/ReadVariableOpReadVariableOp0model_8_dense_103_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'model_8/dense_103/MatMul/ReadVariableOp?
model_8/dense_103/MatMulMatMul$model_8/dropout_45/Identity:output:0/model_8/dense_103/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_8/dense_103/MatMul?
(model_8/dense_103/BiasAdd/ReadVariableOpReadVariableOp1model_8_dense_103_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_8/dense_103/BiasAdd/ReadVariableOp?
model_8/dense_103/BiasAddBiasAdd"model_8/dense_103/MatMul:product:00model_8/dense_103/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_8/dense_103/BiasAdd?
model_8/dense_103/SoftmaxSoftmax"model_8/dense_103/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_8/dense_103/Softmax?
IdentityIdentity#model_8/dense_103/Softmax:softmax:0)^model_8/dense_102/BiasAdd/ReadVariableOp(^model_8/dense_102/MatMul/ReadVariableOp)^model_8/dense_103/BiasAdd/ReadVariableOp(^model_8/dense_103/MatMul/ReadVariableOpE^model_8/token_and_position_embedding_9/embedding_20/embedding_lookupE^model_8/token_and_position_embedding_9/embedding_21/embedding_lookupL^model_8/transformer_block_8/layer_normalization_26/batchnorm/ReadVariableOpP^model_8/transformer_block_8/layer_normalization_26/batchnorm/mul/ReadVariableOpL^model_8/transformer_block_8/layer_normalization_27/batchnorm/ReadVariableOpP^model_8/transformer_block_8/layer_normalization_27/batchnorm/mul/ReadVariableOpS^model_8/transformer_block_8/multi_head_attention_8/dense_96/BiasAdd/ReadVariableOpU^model_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/ReadVariableOpS^model_8/transformer_block_8/multi_head_attention_8/dense_97/BiasAdd/ReadVariableOpU^model_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/ReadVariableOpS^model_8/transformer_block_8/multi_head_attention_8/dense_98/BiasAdd/ReadVariableOpU^model_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/ReadVariableOpS^model_8/transformer_block_8/multi_head_attention_8/dense_99/BiasAdd/ReadVariableOpU^model_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/ReadVariableOpJ^model_8/transformer_block_8/sequential_8/dense_100/BiasAdd/ReadVariableOpL^model_8/transformer_block_8/sequential_8/dense_100/Tensordot/ReadVariableOpJ^model_8/transformer_block_8/sequential_8/dense_101/BiasAdd/ReadVariableOpL^model_8/transformer_block_8/sequential_8/dense_101/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????d: : : : : : : : : : : : : : : : : : : : : : 2T
(model_8/dense_102/BiasAdd/ReadVariableOp(model_8/dense_102/BiasAdd/ReadVariableOp2R
'model_8/dense_102/MatMul/ReadVariableOp'model_8/dense_102/MatMul/ReadVariableOp2T
(model_8/dense_103/BiasAdd/ReadVariableOp(model_8/dense_103/BiasAdd/ReadVariableOp2R
'model_8/dense_103/MatMul/ReadVariableOp'model_8/dense_103/MatMul/ReadVariableOp2?
Dmodel_8/token_and_position_embedding_9/embedding_20/embedding_lookupDmodel_8/token_and_position_embedding_9/embedding_20/embedding_lookup2?
Dmodel_8/token_and_position_embedding_9/embedding_21/embedding_lookupDmodel_8/token_and_position_embedding_9/embedding_21/embedding_lookup2?
Kmodel_8/transformer_block_8/layer_normalization_26/batchnorm/ReadVariableOpKmodel_8/transformer_block_8/layer_normalization_26/batchnorm/ReadVariableOp2?
Omodel_8/transformer_block_8/layer_normalization_26/batchnorm/mul/ReadVariableOpOmodel_8/transformer_block_8/layer_normalization_26/batchnorm/mul/ReadVariableOp2?
Kmodel_8/transformer_block_8/layer_normalization_27/batchnorm/ReadVariableOpKmodel_8/transformer_block_8/layer_normalization_27/batchnorm/ReadVariableOp2?
Omodel_8/transformer_block_8/layer_normalization_27/batchnorm/mul/ReadVariableOpOmodel_8/transformer_block_8/layer_normalization_27/batchnorm/mul/ReadVariableOp2?
Rmodel_8/transformer_block_8/multi_head_attention_8/dense_96/BiasAdd/ReadVariableOpRmodel_8/transformer_block_8/multi_head_attention_8/dense_96/BiasAdd/ReadVariableOp2?
Tmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/ReadVariableOpTmodel_8/transformer_block_8/multi_head_attention_8/dense_96/Tensordot/ReadVariableOp2?
Rmodel_8/transformer_block_8/multi_head_attention_8/dense_97/BiasAdd/ReadVariableOpRmodel_8/transformer_block_8/multi_head_attention_8/dense_97/BiasAdd/ReadVariableOp2?
Tmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/ReadVariableOpTmodel_8/transformer_block_8/multi_head_attention_8/dense_97/Tensordot/ReadVariableOp2?
Rmodel_8/transformer_block_8/multi_head_attention_8/dense_98/BiasAdd/ReadVariableOpRmodel_8/transformer_block_8/multi_head_attention_8/dense_98/BiasAdd/ReadVariableOp2?
Tmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/ReadVariableOpTmodel_8/transformer_block_8/multi_head_attention_8/dense_98/Tensordot/ReadVariableOp2?
Rmodel_8/transformer_block_8/multi_head_attention_8/dense_99/BiasAdd/ReadVariableOpRmodel_8/transformer_block_8/multi_head_attention_8/dense_99/BiasAdd/ReadVariableOp2?
Tmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/ReadVariableOpTmodel_8/transformer_block_8/multi_head_attention_8/dense_99/Tensordot/ReadVariableOp2?
Imodel_8/transformer_block_8/sequential_8/dense_100/BiasAdd/ReadVariableOpImodel_8/transformer_block_8/sequential_8/dense_100/BiasAdd/ReadVariableOp2?
Kmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/ReadVariableOpKmodel_8/transformer_block_8/sequential_8/dense_100/Tensordot/ReadVariableOp2?
Imodel_8/transformer_block_8/sequential_8/dense_101/BiasAdd/ReadVariableOpImodel_8/transformer_block_8/sequential_8/dense_101/BiasAdd/ReadVariableOp2?
Kmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/ReadVariableOpKmodel_8/transformer_block_8/sequential_8/dense_101/Tensordot/ReadVariableOp:Q M
'
_output_shapes
:?????????d
"
_user_specified_name
input_11
?
?
)__inference_dense_100_layer_call_fn_93793

inputs
unknown:@ 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_100_layer_call_and_return_conditional_losses_908492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????d 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????d@: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????d@
 
_user_specified_nameinputs
? 
?
D__inference_dense_101_layer_call_and_return_conditional_losses_93863

inputs3
!tensordot_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????d 2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????d@2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????d : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:?????????d 
 
_user_specified_nameinputs
?
?
G__inference_sequential_8_layer_call_and_return_conditional_losses_90990
dense_100_input!
dense_100_90979:@ 
dense_100_90981: !
dense_101_90984: @
dense_101_90986:@
identity??!dense_100/StatefulPartitionedCall?!dense_101/StatefulPartitionedCall?
!dense_100/StatefulPartitionedCallStatefulPartitionedCalldense_100_inputdense_100_90979dense_100_90981*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_100_layer_call_and_return_conditional_losses_908492#
!dense_100/StatefulPartitionedCall?
!dense_101/StatefulPartitionedCallStatefulPartitionedCall*dense_100/StatefulPartitionedCall:output:0dense_101_90984dense_101_90986*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_101_layer_call_and_return_conditional_losses_908852#
!dense_101/StatefulPartitionedCall?
IdentityIdentity*dense_101/StatefulPartitionedCall:output:0"^dense_100/StatefulPartitionedCall"^dense_101/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????d@: : : : 2F
!dense_100/StatefulPartitionedCall!dense_100/StatefulPartitionedCall2F
!dense_101/StatefulPartitionedCall!dense_101/StatefulPartitionedCall:\ X
+
_output_shapes
:?????????d@
)
_user_specified_namedense_100_input
? 
?
D__inference_dense_101_layer_call_and_return_conditional_losses_90885

inputs3
!tensordot_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: @*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????d 2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????d@2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????d@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:?????????d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????d : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:?????????d 
 
_user_specified_nameinputs
?
?
G__inference_sequential_8_layer_call_and_return_conditional_losses_91004
dense_100_input!
dense_100_90993:@ 
dense_100_90995: !
dense_101_90998: @
dense_101_91000:@
identity??!dense_100/StatefulPartitionedCall?!dense_101/StatefulPartitionedCall?
!dense_100/StatefulPartitionedCallStatefulPartitionedCalldense_100_inputdense_100_90993dense_100_90995*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_100_layer_call_and_return_conditional_losses_908492#
!dense_100/StatefulPartitionedCall?
!dense_101/StatefulPartitionedCallStatefulPartitionedCall*dense_100/StatefulPartitionedCall:output:0dense_101_90998dense_101_91000*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_101_layer_call_and_return_conditional_losses_908852#
!dense_101/StatefulPartitionedCall?
IdentityIdentity*dense_101/StatefulPartitionedCall:output:0"^dense_100/StatefulPartitionedCall"^dense_101/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????d@: : : : 2F
!dense_100/StatefulPartitionedCall!dense_100/StatefulPartitionedCall2F
!dense_101/StatefulPartitionedCall!dense_101/StatefulPartitionedCall:\ X
+
_output_shapes
:?????????d@
)
_user_specified_namedense_100_input
?,
?	
B__inference_model_8_layer_call_and_return_conditional_losses_92117
input_116
$token_and_position_embedding_9_92065:d@7
$token_and_position_embedding_9_92067:	?@@+
transformer_block_8_92070:@@'
transformer_block_8_92072:@+
transformer_block_8_92074:@@'
transformer_block_8_92076:@+
transformer_block_8_92078:@@'
transformer_block_8_92080:@+
transformer_block_8_92082:@@'
transformer_block_8_92084:@'
transformer_block_8_92086:@'
transformer_block_8_92088:@+
transformer_block_8_92090:@ '
transformer_block_8_92092: +
transformer_block_8_92094: @'
transformer_block_8_92096:@'
transformer_block_8_92098:@'
transformer_block_8_92100:@!
dense_102_92105:@
dense_102_92107:!
dense_103_92111:
dense_103_92113:
identity??!dense_102/StatefulPartitionedCall?!dense_103/StatefulPartitionedCall?6token_and_position_embedding_9/StatefulPartitionedCall?+transformer_block_8/StatefulPartitionedCall?
6token_and_position_embedding_9/StatefulPartitionedCallStatefulPartitionedCallinput_11$token_and_position_embedding_9_92065$token_and_position_embedding_9_92067*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *b
f]R[
Y__inference_token_and_position_embedding_9_layer_call_and_return_conditional_losses_9105928
6token_and_position_embedding_9/StatefulPartitionedCall?
+transformer_block_8/StatefulPartitionedCallStatefulPartitionedCall?token_and_position_embedding_9/StatefulPartitionedCall:output:0transformer_block_8_92070transformer_block_8_92072transformer_block_8_92074transformer_block_8_92076transformer_block_8_92078transformer_block_8_92080transformer_block_8_92082transformer_block_8_92084transformer_block_8_92086transformer_block_8_92088transformer_block_8_92090transformer_block_8_92092transformer_block_8_92094transformer_block_8_92096transformer_block_8_92098transformer_block_8_92100*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????d@*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_transformer_block_8_layer_call_and_return_conditional_losses_913092-
+transformer_block_8/StatefulPartitionedCall?
*global_average_pooling1d_8/PartitionedCallPartitionedCall4transformer_block_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_913482,
*global_average_pooling1d_8/PartitionedCall?
dropout_44/PartitionedCallPartitionedCall3global_average_pooling1d_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_44_layer_call_and_return_conditional_losses_913552
dropout_44/PartitionedCall?
!dense_102/StatefulPartitionedCallStatefulPartitionedCall#dropout_44/PartitionedCall:output:0dense_102_92105dense_102_92107*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_102_layer_call_and_return_conditional_losses_913682#
!dense_102/StatefulPartitionedCall?
dropout_45/PartitionedCallPartitionedCall*dense_102/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_45_layer_call_and_return_conditional_losses_913792
dropout_45/PartitionedCall?
!dense_103/StatefulPartitionedCallStatefulPartitionedCall#dropout_45/PartitionedCall:output:0dense_103_92111dense_103_92113*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_103_layer_call_and_return_conditional_losses_913922#
!dense_103/StatefulPartitionedCall?
IdentityIdentity*dense_103/StatefulPartitionedCall:output:0"^dense_102/StatefulPartitionedCall"^dense_103/StatefulPartitionedCall7^token_and_position_embedding_9/StatefulPartitionedCall,^transformer_block_8/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????d: : : : : : : : : : : : : : : : : : : : : : 2F
!dense_102/StatefulPartitionedCall!dense_102/StatefulPartitionedCall2F
!dense_103/StatefulPartitionedCall!dense_103/StatefulPartitionedCall2p
6token_and_position_embedding_9/StatefulPartitionedCall6token_and_position_embedding_9/StatefulPartitionedCall2Z
+transformer_block_8/StatefulPartitionedCall+transformer_block_8/StatefulPartitionedCall:Q M
'
_output_shapes
:?????????d
"
_user_specified_name
input_11"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
=
input_111
serving_default_input_11:0?????????d=
	dense_1030
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
? 
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
		optimizer

trainable_variables
regularization_losses
	variables
	keras_api

signatures
?_default_save_signature
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_network?{"name": "model_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model_8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_11"}, "name": "input_11", "inbound_nodes": []}, {"class_name": "TokenAndPositionEmbedding", "config": {"layer was saved without config": true}, "name": "token_and_position_embedding_9", "inbound_nodes": [[["input_11", 0, 0, {}]]]}, {"class_name": "TransformerBlock", "config": {"layer was saved without config": true}, "name": "transformer_block_8", "inbound_nodes": [[["token_and_position_embedding_9", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling1D", "config": {"name": "global_average_pooling1d_8", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling1d_8", "inbound_nodes": [[["transformer_block_8", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_44", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_44", "inbound_nodes": [[["global_average_pooling1d_8", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_102", "trainable": true, "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_102", "inbound_nodes": [[["dropout_44", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_45", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_45", "inbound_nodes": [[["dense_102", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_103", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_103", "inbound_nodes": [[["dropout_45", 0, 0, {}]]]}], "input_layers": [["input_11", 0, 0]], "output_layers": [["dense_103", 0, 0]]}, "shared_object_id": 10, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 100]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 100]}, "float32", "input_11"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}, "shared_object_id": 12}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_11", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_11"}}
?
	token_emb
pos_emb
trainable_variables
regularization_losses
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "token_and_position_embedding_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "TokenAndPositionEmbedding", "config": {"layer was saved without config": true}}
?
att
ffn

layernorm1

layernorm2
dropout1
dropout2
trainable_variables
regularization_losses
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "transformer_block_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "TransformerBlock", "config": {"layer was saved without config": true}}
?
trainable_variables
 regularization_losses
!	variables
"	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "global_average_pooling1d_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "GlobalAveragePooling1D", "config": {"name": "global_average_pooling1d_8", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["transformer_block_8", 0, 0, {}]]], "shared_object_id": 1, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 13}}
?
#trainable_variables
$regularization_losses
%	variables
&	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dropout_44", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_44", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "inbound_nodes": [[["global_average_pooling1d_8", 0, 0, {}]]], "shared_object_id": 2}
?	

'kernel
(bias
)trainable_variables
*regularization_losses
+	variables
,	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_102", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_102", "trainable": true, "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 3}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout_44", 0, 0, {}]]], "shared_object_id": 5, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 14}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?
-trainable_variables
.regularization_losses
/	variables
0	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dropout_45", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_45", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "inbound_nodes": [[["dense_102", 0, 0, {}]]], "shared_object_id": 6}
?	

1kernel
2bias
3trainable_variables
4regularization_losses
5	variables
6	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_103", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_103", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout_45", 0, 0, {}]]], "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}, "shared_object_id": 15}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20]}}
?
7iter

8beta_1

9beta_2
	:decay
;learning_rate'm?(m?1m?2m?<m?=m?>m??m?@m?Am?Bm?Cm?Dm?Em?Fm?Gm?Hm?Im?Jm?Km?Lm?Mm?'v?(v?1v?2v?<v?=v?>v??v?@v?Av?Bv?Cv?Dv?Ev?Fv?Gv?Hv?Iv?Jv?Kv?Lv?Mv?"
	optimizer
?
<0
=1
>2
?3
@4
A5
B6
C7
D8
E9
F10
G11
H12
I13
J14
K15
L16
M17
'18
(19
120
221"
trackable_list_wrapper
 "
trackable_list_wrapper
?
<0
=1
>2
?3
@4
A5
B6
C7
D8
E9
F10
G11
H12
I13
J14
K15
L16
M17
'18
(19
120
221"
trackable_list_wrapper
?
Nlayer_regularization_losses

Olayers

trainable_variables
regularization_losses
	variables
Pmetrics
Qnon_trainable_variables
Rlayer_metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?
<
embeddings
Strainable_variables
Tregularization_losses
U	variables
V	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "embedding_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "class_name": "Embedding", "config": {"name": "embedding_20", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 8192, "output_dim": 64, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 16}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "shared_object_id": 17, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
?
=
embeddings
Wtrainable_variables
Xregularization_losses
Y	variables
Z	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "embedding_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "class_name": "Embedding", "config": {"name": "embedding_21", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 100, "output_dim": 64, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 18}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "shared_object_id": 19, "build_input_shape": {"class_name": "TensorShape", "items": [null]}}
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
?
[layer_regularization_losses

\layers
trainable_variables
regularization_losses
	variables
]metrics
^non_trainable_variables
_layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
`query_dense
a	key_dense
bvalue_dense
	cdense
dtrainable_variables
eregularization_losses
f	variables
g	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "multi_head_attention_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MultiHeadAttention", "config": {"layer was saved without config": true}}
?
hlayer_with_weights-0
hlayer-0
ilayer_with_weights-1
ilayer-1
jtrainable_variables
kregularization_losses
l	variables
m	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_sequential?{"name": "sequential_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_100_input"}}, {"class_name": "Dense", "config": {"name": "dense_100", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_101", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 27, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 28}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 64]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 100, 64]}, "float32", "dense_100_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_100_input"}, "shared_object_id": 20}, {"class_name": "Dense", "config": {"name": "dense_100", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 21}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 23}, {"class_name": "Dense", "config": {"name": "dense_101", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 24}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 26}]}}}
?
naxis
	Jgamma
Kbeta
otrainable_variables
pregularization_losses
q	variables
r	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "layer_normalization_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LayerNormalization", "config": {"name": "layer_normalization_26", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 29}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 30}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 31, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 64]}}
?
saxis
	Lgamma
Mbeta
ttrainable_variables
uregularization_losses
v	variables
w	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "layer_normalization_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LayerNormalization", "config": {"name": "layer_normalization_27", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 32}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 33}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 34, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 64]}}
?
xtrainable_variables
yregularization_losses
z	variables
{	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dropout_42", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_42", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 35}
?
|trainable_variables
}regularization_losses
~	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dropout_43", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_43", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 36}
?
>0
?1
@2
A3
B4
C5
D6
E7
F8
G9
H10
I11
J12
K13
L14
M15"
trackable_list_wrapper
 "
trackable_list_wrapper
?
>0
?1
@2
A3
B4
C5
D6
E7
F8
G9
H10
I11
J12
K13
L14
M15"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
trainable_variables
regularization_losses
	variables
?metrics
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
trainable_variables
 regularization_losses
!	variables
?metrics
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
#trainable_variables
$regularization_losses
%	variables
?metrics
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": @2dense_102/kernel
:2dense_102/bias
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
)trainable_variables
*regularization_losses
+	variables
?metrics
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
-trainable_variables
.regularization_losses
/	variables
?metrics
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 2dense_103/kernel
:2dense_103/bias
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
3trainable_variables
4regularization_losses
5	variables
?metrics
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
I:G	?@@26token_and_position_embedding_9/embedding_20/embeddings
H:Fd@26token_and_position_embedding_9/embedding_21/embeddings
L:J@@2:transformer_block_8/multi_head_attention_8/dense_96/kernel
F:D@28transformer_block_8/multi_head_attention_8/dense_96/bias
L:J@@2:transformer_block_8/multi_head_attention_8/dense_97/kernel
F:D@28transformer_block_8/multi_head_attention_8/dense_97/bias
L:J@@2:transformer_block_8/multi_head_attention_8/dense_98/kernel
F:D@28transformer_block_8/multi_head_attention_8/dense_98/bias
L:J@@2:transformer_block_8/multi_head_attention_8/dense_99/kernel
F:D@28transformer_block_8/multi_head_attention_8/dense_99/bias
": @ 2dense_100/kernel
: 2dense_100/bias
":  @2dense_101/kernel
:@2dense_101/bias
>:<@20transformer_block_8/layer_normalization_26/gamma
=:;@2/transformer_block_8/layer_normalization_26/beta
>:<@20transformer_block_8/layer_normalization_27/gamma
=:;@2/transformer_block_8/layer_normalization_27/beta
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
<0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
<0"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
Strainable_variables
Tregularization_losses
U	variables
?metrics
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
=0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
=0"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
Wtrainable_variables
Xregularization_losses
Y	variables
?metrics
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?

>kernel
?bias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_96", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_96", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 37}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 38}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 39, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 40}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 64]}}
?

@kernel
Abias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_97", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_97", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 41}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 42}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 43, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 44}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 64]}}
?

Bkernel
Cbias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_98", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_98", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 45}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 46}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 47, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 48}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 64]}}
?

Dkernel
Ebias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_99", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_99", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 49}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 50}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 51, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 52}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 64]}}
X
>0
?1
@2
A3
B4
C5
D6
E7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
>0
?1
@2
A3
B4
C5
D6
E7"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
dtrainable_variables
eregularization_losses
f	variables
?metrics
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

Fkernel
Gbias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_100", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_100", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 21}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 23, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 28}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 64]}}
?

Hkernel
Ibias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_101", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_101", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 24}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 26, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 53}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 32]}}
<
F0
G1
H2
I3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
F0
G1
H2
I3"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
jtrainable_variables
kregularization_losses
l	variables
?metrics
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
otrainable_variables
pregularization_losses
q	variables
?metrics
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
ttrainable_variables
uregularization_losses
v	variables
?metrics
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
xtrainable_variables
yregularization_losses
z	variables
?metrics
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
|trainable_variables
}regularization_losses
~	variables
?metrics
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 54}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}, "shared_object_id": 12}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
?	variables
?metrics
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
?	variables
?metrics
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
?	variables
?metrics
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
?	variables
?metrics
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
`0
a1
b2
c3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
?	variables
?metrics
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
?
 ?layer_regularization_losses
?layers
?trainable_variables
?regularization_losses
?	variables
?metrics
?non_trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
':%@2Adam/dense_102/kernel/m
!:2Adam/dense_102/bias/m
':%2Adam/dense_103/kernel/m
!:2Adam/dense_103/bias/m
N:L	?@@2=Adam/token_and_position_embedding_9/embedding_20/embeddings/m
M:Kd@2=Adam/token_and_position_embedding_9/embedding_21/embeddings/m
Q:O@@2AAdam/transformer_block_8/multi_head_attention_8/dense_96/kernel/m
K:I@2?Adam/transformer_block_8/multi_head_attention_8/dense_96/bias/m
Q:O@@2AAdam/transformer_block_8/multi_head_attention_8/dense_97/kernel/m
K:I@2?Adam/transformer_block_8/multi_head_attention_8/dense_97/bias/m
Q:O@@2AAdam/transformer_block_8/multi_head_attention_8/dense_98/kernel/m
K:I@2?Adam/transformer_block_8/multi_head_attention_8/dense_98/bias/m
Q:O@@2AAdam/transformer_block_8/multi_head_attention_8/dense_99/kernel/m
K:I@2?Adam/transformer_block_8/multi_head_attention_8/dense_99/bias/m
':%@ 2Adam/dense_100/kernel/m
!: 2Adam/dense_100/bias/m
':% @2Adam/dense_101/kernel/m
!:@2Adam/dense_101/bias/m
C:A@27Adam/transformer_block_8/layer_normalization_26/gamma/m
B:@@26Adam/transformer_block_8/layer_normalization_26/beta/m
C:A@27Adam/transformer_block_8/layer_normalization_27/gamma/m
B:@@26Adam/transformer_block_8/layer_normalization_27/beta/m
':%@2Adam/dense_102/kernel/v
!:2Adam/dense_102/bias/v
':%2Adam/dense_103/kernel/v
!:2Adam/dense_103/bias/v
N:L	?@@2=Adam/token_and_position_embedding_9/embedding_20/embeddings/v
M:Kd@2=Adam/token_and_position_embedding_9/embedding_21/embeddings/v
Q:O@@2AAdam/transformer_block_8/multi_head_attention_8/dense_96/kernel/v
K:I@2?Adam/transformer_block_8/multi_head_attention_8/dense_96/bias/v
Q:O@@2AAdam/transformer_block_8/multi_head_attention_8/dense_97/kernel/v
K:I@2?Adam/transformer_block_8/multi_head_attention_8/dense_97/bias/v
Q:O@@2AAdam/transformer_block_8/multi_head_attention_8/dense_98/kernel/v
K:I@2?Adam/transformer_block_8/multi_head_attention_8/dense_98/bias/v
Q:O@@2AAdam/transformer_block_8/multi_head_attention_8/dense_99/kernel/v
K:I@2?Adam/transformer_block_8/multi_head_attention_8/dense_99/bias/v
':%@ 2Adam/dense_100/kernel/v
!: 2Adam/dense_100/bias/v
':% @2Adam/dense_101/kernel/v
!:@2Adam/dense_101/bias/v
C:A@27Adam/transformer_block_8/layer_normalization_26/gamma/v
B:@@26Adam/transformer_block_8/layer_normalization_26/beta/v
C:A@27Adam/transformer_block_8/layer_normalization_27/gamma/v
B:@@26Adam/transformer_block_8/layer_normalization_27/beta/v
?2?
 __inference__wrapped_model_90811?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *'?$
"?
input_11?????????d
?2?
'__inference_model_8_layer_call_fn_91446
'__inference_model_8_layer_call_fn_92278
'__inference_model_8_layer_call_fn_92327
'__inference_model_8_layer_call_fn_92062?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_model_8_layer_call_and_return_conditional_losses_92609
B__inference_model_8_layer_call_and_return_conditional_losses_92919
B__inference_model_8_layer_call_and_return_conditional_losses_92117
B__inference_model_8_layer_call_and_return_conditional_losses_92172?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
>__inference_token_and_position_embedding_9_layer_call_fn_92928?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
Y__inference_token_and_position_embedding_9_layer_call_and_return_conditional_losses_92952?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
3__inference_transformer_block_8_layer_call_fn_92989
3__inference_transformer_block_8_layer_call_fn_93026?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults? 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
N__inference_transformer_block_8_layer_call_and_return_conditional_losses_93270
N__inference_transformer_block_8_layer_call_and_return_conditional_losses_93528?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults? 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
:__inference_global_average_pooling1d_8_layer_call_fn_93533
:__inference_global_average_pooling1d_8_layer_call_fn_93538?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
U__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_93544
U__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_93550?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dropout_44_layer_call_fn_93555
*__inference_dropout_44_layer_call_fn_93560?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_44_layer_call_and_return_conditional_losses_93565
E__inference_dropout_44_layer_call_and_return_conditional_losses_93577?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dense_102_layer_call_fn_93586?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_102_layer_call_and_return_conditional_losses_93597?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dropout_45_layer_call_fn_93602
*__inference_dropout_45_layer_call_fn_93607?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_45_layer_call_and_return_conditional_losses_93612
E__inference_dropout_45_layer_call_and_return_conditional_losses_93624?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dense_103_layer_call_fn_93633?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_103_layer_call_and_return_conditional_losses_93644?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_92229input_11"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_sequential_8_layer_call_fn_90903
,__inference_sequential_8_layer_call_fn_93657
,__inference_sequential_8_layer_call_fn_93670
,__inference_sequential_8_layer_call_fn_90976?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_sequential_8_layer_call_and_return_conditional_losses_93727
G__inference_sequential_8_layer_call_and_return_conditional_losses_93784
G__inference_sequential_8_layer_call_and_return_conditional_losses_90990
G__inference_sequential_8_layer_call_and_return_conditional_losses_91004?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_100_layer_call_fn_93793?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_100_layer_call_and_return_conditional_losses_93824?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_101_layer_call_fn_93833?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_101_layer_call_and_return_conditional_losses_93863?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_90811?=<>?@ABCDEJKFGHILM'(121?.
'?$
"?
input_11?????????d
? "5?2
0
	dense_103#? 
	dense_103??????????
D__inference_dense_100_layer_call_and_return_conditional_losses_93824dFG3?0
)?&
$?!
inputs?????????d@
? ")?&
?
0?????????d 
? ?
)__inference_dense_100_layer_call_fn_93793WFG3?0
)?&
$?!
inputs?????????d@
? "??????????d ?
D__inference_dense_101_layer_call_and_return_conditional_losses_93863dHI3?0
)?&
$?!
inputs?????????d 
? ")?&
?
0?????????d@
? ?
)__inference_dense_101_layer_call_fn_93833WHI3?0
)?&
$?!
inputs?????????d 
? "??????????d@?
D__inference_dense_102_layer_call_and_return_conditional_losses_93597\'(/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? |
)__inference_dense_102_layer_call_fn_93586O'(/?,
%?"
 ?
inputs?????????@
? "???????????
D__inference_dense_103_layer_call_and_return_conditional_losses_93644\12/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? |
)__inference_dense_103_layer_call_fn_93633O12/?,
%?"
 ?
inputs?????????
? "???????????
E__inference_dropout_44_layer_call_and_return_conditional_losses_93565\3?0
)?&
 ?
inputs?????????@
p 
? "%?"
?
0?????????@
? ?
E__inference_dropout_44_layer_call_and_return_conditional_losses_93577\3?0
)?&
 ?
inputs?????????@
p
? "%?"
?
0?????????@
? }
*__inference_dropout_44_layer_call_fn_93555O3?0
)?&
 ?
inputs?????????@
p 
? "??????????@}
*__inference_dropout_44_layer_call_fn_93560O3?0
)?&
 ?
inputs?????????@
p
? "??????????@?
E__inference_dropout_45_layer_call_and_return_conditional_losses_93612\3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
E__inference_dropout_45_layer_call_and_return_conditional_losses_93624\3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? }
*__inference_dropout_45_layer_call_fn_93602O3?0
)?&
 ?
inputs?????????
p 
? "??????????}
*__inference_dropout_45_layer_call_fn_93607O3?0
)?&
 ?
inputs?????????
p
? "???????????
U__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_93544{I?F
??<
6?3
inputs'???????????????????????????

 
? ".?+
$?!
0??????????????????
? ?
U__inference_global_average_pooling1d_8_layer_call_and_return_conditional_losses_93550`7?4
-?*
$?!
inputs?????????d@

 
? "%?"
?
0?????????@
? ?
:__inference_global_average_pooling1d_8_layer_call_fn_93533nI?F
??<
6?3
inputs'???????????????????????????

 
? "!????????????????????
:__inference_global_average_pooling1d_8_layer_call_fn_93538S7?4
-?*
$?!
inputs?????????d@

 
? "??????????@?
B__inference_model_8_layer_call_and_return_conditional_losses_92117z=<>?@ABCDEJKFGHILM'(129?6
/?,
"?
input_11?????????d
p 

 
? "%?"
?
0?????????
? ?
B__inference_model_8_layer_call_and_return_conditional_losses_92172z=<>?@ABCDEJKFGHILM'(129?6
/?,
"?
input_11?????????d
p

 
? "%?"
?
0?????????
? ?
B__inference_model_8_layer_call_and_return_conditional_losses_92609x=<>?@ABCDEJKFGHILM'(127?4
-?*
 ?
inputs?????????d
p 

 
? "%?"
?
0?????????
? ?
B__inference_model_8_layer_call_and_return_conditional_losses_92919x=<>?@ABCDEJKFGHILM'(127?4
-?*
 ?
inputs?????????d
p

 
? "%?"
?
0?????????
? ?
'__inference_model_8_layer_call_fn_91446m=<>?@ABCDEJKFGHILM'(129?6
/?,
"?
input_11?????????d
p 

 
? "???????????
'__inference_model_8_layer_call_fn_92062m=<>?@ABCDEJKFGHILM'(129?6
/?,
"?
input_11?????????d
p

 
? "???????????
'__inference_model_8_layer_call_fn_92278k=<>?@ABCDEJKFGHILM'(127?4
-?*
 ?
inputs?????????d
p 

 
? "???????????
'__inference_model_8_layer_call_fn_92327k=<>?@ABCDEJKFGHILM'(127?4
-?*
 ?
inputs?????????d
p

 
? "???????????
G__inference_sequential_8_layer_call_and_return_conditional_losses_90990wFGHID?A
:?7
-?*
dense_100_input?????????d@
p 

 
? ")?&
?
0?????????d@
? ?
G__inference_sequential_8_layer_call_and_return_conditional_losses_91004wFGHID?A
:?7
-?*
dense_100_input?????????d@
p

 
? ")?&
?
0?????????d@
? ?
G__inference_sequential_8_layer_call_and_return_conditional_losses_93727nFGHI;?8
1?.
$?!
inputs?????????d@
p 

 
? ")?&
?
0?????????d@
? ?
G__inference_sequential_8_layer_call_and_return_conditional_losses_93784nFGHI;?8
1?.
$?!
inputs?????????d@
p

 
? ")?&
?
0?????????d@
? ?
,__inference_sequential_8_layer_call_fn_90903jFGHID?A
:?7
-?*
dense_100_input?????????d@
p 

 
? "??????????d@?
,__inference_sequential_8_layer_call_fn_90976jFGHID?A
:?7
-?*
dense_100_input?????????d@
p

 
? "??????????d@?
,__inference_sequential_8_layer_call_fn_93657aFGHI;?8
1?.
$?!
inputs?????????d@
p 

 
? "??????????d@?
,__inference_sequential_8_layer_call_fn_93670aFGHI;?8
1?.
$?!
inputs?????????d@
p

 
? "??????????d@?
#__inference_signature_wrapper_92229?=<>?@ABCDEJKFGHILM'(12=?:
? 
3?0
.
input_11"?
input_11?????????d"5?2
0
	dense_103#? 
	dense_103??????????
Y__inference_token_and_position_embedding_9_layer_call_and_return_conditional_losses_92952[=<*?'
 ?
?
x?????????d
? ")?&
?
0?????????d@
? ?
>__inference_token_and_position_embedding_9_layer_call_fn_92928N=<*?'
 ?
?
x?????????d
? "??????????d@?
N__inference_transformer_block_8_layer_call_and_return_conditional_losses_93270v>?@ABCDEJKFGHILM7?4
-?*
$?!
inputs?????????d@
p 
? ")?&
?
0?????????d@
? ?
N__inference_transformer_block_8_layer_call_and_return_conditional_losses_93528v>?@ABCDEJKFGHILM7?4
-?*
$?!
inputs?????????d@
p
? ")?&
?
0?????????d@
? ?
3__inference_transformer_block_8_layer_call_fn_92989i>?@ABCDEJKFGHILM7?4
-?*
$?!
inputs?????????d@
p 
? "??????????d@?
3__inference_transformer_block_8_layer_call_fn_93026i>?@ABCDEJKFGHILM7?4
-?*
$?!
inputs?????????d@
p
? "??????????d@