import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib import pyplot as plt
import random
from sklearn import linear_model
import seaborn as sns


#obtain data source
#seqs
x_data0to29=np.loadtxt('/Users/wenq10/hpmt/generated_sqs/seq0-29.csv',delimiter=",")
x_data30to100=np.loadtxt('/Users/wenq10/hpmt/generated_sqs/seq30-100.csv',delimiter=",")
x_data100to255=np.loadtxt('/Users/wenq10/hpmt/generated_sqs/seq100-255.csv',delimiter=",")

x_data0to29=x_data0to29.reshape([-1,24,13])
x_data30to100=x_data30to100.reshape([-1,24,13])
x_data100to255=x_data100to255.reshape([-1,24,13])

x_data=np.vstack([x_data0to29,x_data30to100,x_data100to255])
x_data=x_data.astype(np.float32)
#y
y_data0to29=np.loadtxt('/Users/wenq10/hpmt/generated_sqs/y0-29.csv',delimiter=",")
y_data30to100=np.loadtxt('/Users/wenq10/hpmt/generated_sqs/y30-100.csv',delimiter=",")
y_data100to255=np.loadtxt('/Users/wenq10/hpmt/generated_sqs/y100-255.csv',delimiter=",")

y_data=np.vstack([y_data0to29,y_data30to100,y_data100to255])

y_data=y_data.astype(np.float32)

y_data=y_data[:,[0,2,3,4,5,6,7,8,9,10,11,12]]
#split into train-test




# hyperparameters
lr = 0.001
#training_iters = 100000
batch_size = 100

n_inputs = 13
n_steps = 24
n_hidden_units = 128
n_outputs=12

X_dta = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y_dta = tf.placeholder(tf.float32, [None, n_outputs])

weights = {
	# (13, 128)
	'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
	# (128, 13)
	'out': tf.Variable(tf.random_normal([n_hidden_units, n_outputs]))
}

biases = {
	# (128, )
	'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
	# (13, )
	'out': tf.Variable(tf.constant(0.1, shape=[n_outputs, ]))
}

X_dta_flt = tf.reshape(X_dta, [-1, n_inputs])
X_dta_flt_in = tf.matmul(X_dta_flt, weights['in']) + biases['in']
X_dta_in = tf.reshape(X_dta_flt_in, [-1, n_steps, n_hidden_units])

cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
init_state = cell.zero_state(batch_size, dtype=tf.float32)

outputs, final_state = tf.nn.dynamic_rnn(cell, X_dta_in, initial_state=init_state, time_major=False)
outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
y_hat = tf.matmul(outputs[-1], weights['out']) + biases['out']	# shape = (128, 12)

cost = tf.reduce_mean(tf.reduce_mean(tf.square(y_dta-y_hat),axis=1))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)
init = tf.global_variables_initializer()

sess=tf.Session()
sess.run(init)

#start training, inputs are x_data & y_data
for step_i in range(30000):
	sess.run(train_op, feed_dict={
		X_dta: x_data[((step_i%800)*batch_size):(((step_i%800)*batch_size)+batch_size),:,:],
		y_dta: y_data[((step_i%800)*batch_size):(((step_i%800)*batch_size)+batch_size),:]
		})
	if (step_i%1000==0):
		print(sess.run(cost, feed_dict={
			X_dta: x_data[((step_i%800)*batch_size):(((step_i%800)*batch_size)+batch_size),:,:],
			y_dta: y_data[((step_i%800)*batch_size):(((step_i%800)*batch_size)+batch_size),:]
		}))

#evaluate autoregression outcome: training error
y_dta_ht=[]
for rslt_i in range(800):
	y_dta_ht.append(sess.run(y_hat, feed_dict={
			X_dta: x_data[((rslt_i%800)*batch_size):(((rslt_i%800)*batch_size)+batch_size),:,:],
			y_dta: y_data[((rslt_i%800)*batch_size):(((rslt_i%800)*batch_size)+batch_size),:]
		}))

#transform
y_dta_ht=np.array(y_dta_ht).reshape([-1,n_outputs])
#38400=128*300
np.corrcoef(y_dta_ht[:,0],y_data[0:80000,0])


#held-out set evaluation: test error
y_ho_hat=[]
for ho_i in range(101):
	y_ho_hat.append(sess.run(y_hat, feed_dict={
			X_dta: x_data[(80000+ho_i*batch_size):((80000+ho_i*batch_size)+batch_size),:,:],
			y_dta: y_data[(80000+ho_i*batch_size):((80000+ho_i*batch_size)+batch_size),:]
		}))
y_ho_hat=np.array(y_ho_hat).reshape([-1,n_outputs])

y_ho_data=y_data[80000:90100,:]

np.corrcoef(y_ho_hat[:,11],y_ho_data[:,11])


#visualize autoregression prediction
heatmap, xedges, yedges = np.histogram2d(y_ho_hat[:,3], y_ho_data[:,3], bins=100)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
#extent=[0,1,0,1]
trans2blue = colors.LinearSegmentedColormap.from_list(name='Trans2Blue', colors=[(0., 0., 1., 0.), (0., 0., 1., 1.)])
plt.imshow(heatmap.T, extent=extent, origin='lower',cmap=trans2blue)
plt.show()

plt.scatter(y_ho_hat[:,12],y_ho_data[:,12])

#====================================use trained cell state for prediction=============================================
#read personality data
psnlty=pd.read_csv('/Users/wenq10/hpmt/generated_sqs/user_psnlty.csv',sep=';')
psnlty.drop(['id'],axis=1,inplace=True)
#read corresponding userinfo (as the bridge between seqs and psnlty)
user_info=pd.read_csv('/Users/wenq10/hpmt/generated_sqs/usrinfo.csv',sep=',')
user_info.drop(['Unnamed: 0'],axis=1,inplace=True)
user_info=user_info.iloc[np.argwhere(user_info.nb_seqs>0).reshape(-1),]

sum(user_info.nb_seqs.iloc[0:30])

#users with personality data
urs_with_psnlty=psnlty.user_id.drop_duplicates().tolist()
#users with both sensor data and personality data
usrs_with_both=[ur for ur in urs_with_psnlty if ur in user_info.user_id.tolist()]

usrs_data_range=np.zeros([len(usrs_with_both),2])
for ur_i in range(len(usrs_with_both)):
	usr_pos=np.argwhere(user_info.user_id==usrs_with_both[ur_i])[0][0]
	tempt_start=sum(user_info.nb_seqs.iloc[0:usr_pos])
	tempt_end=sum(user_info.nb_seqs.iloc[0:(usr_pos+1)])
	usrs_data_range[ur_i,:]=np.array([tempt_start,tempt_end])

usrs_data_range=usrs_data_range.astype(np.int)
#collect samples to calculate cell state
sample_indices=np.zeros([usrs_data_range.shape[0],batch_size])
for row_i in range(usrs_data_range.shape[0]):
	sample_indices[row_i,:] = np.array([random.sample(range(usrs_data_range[row_i,0],usrs_data_range[row_i,1]), 1) for _ in range(100)]).reshape(-1)

sample_indices=sample_indices.astype(np.int)

#estimate long term cell state
cell_state=np.zeros([usrs_data_range.shape[0],n_hidden_units])
for ur_i in range(usrs_data_range.shape[0]):
	cstate_tempt=sess.run(final_state, feed_dict={
			X_dta: x_data[sample_indices[ur_i]],
			y_dta: y_data[sample_indices[ur_i]]
		})[0]
	cell_state[ur_i,:]=np.mean(cstate_tempt,axis=0)

#normalize cell_state: not necessary
for col_c in range(cell_state.shape[1]):
	tempt_col=(cell_state[:,col_c]-np.mean(cell_state[:,col_c]))/np.std(cell_state[:,col_c])
	cell_state[:,col_c]=tempt_col

np.savetxt('/Users/wenq10/hpmt/generated_sqs/cell_states46.csv', cell_state, delimiter = ',')
#cell_state=np.loadtxt('/Users/wenq10/hpmt/generated_sqs/cell_states46.csv',delimiter=",")


#===================================correlation and lasso=========================================
psnlty_usable=np.array([psnlty[psnlty.user_id==ur].iloc[0,1:].tolist() for ur in usrs_with_both])
for col_p in range(psnlty_usable.shape[1]):
	tempt_col=(psnlty_usable[:,col_p]-np.mean(psnlty_usable[:,col_p]))/np.std(psnlty_usable[:,col_p])
	psnlty_usable[:,col_p]=tempt_col


DI_cor_mat=np.array(pd.DataFrame(np.hstack([cell_state,psnlty_usable])).corr())[128:133,0:128]
sns.heatmap(DI_cor_mat)
plt.yticks(np.array(range(psnlty_usable.shape[1]))+0.5,['C','A','O','E','N'])
plt.xticks([])
plt.show()


#lasso to pick up the best feature
las_model = linear_model.Lasso(alpha=0.1)

psnlty_hat=np.zeros(psnlty_usable.shape)
R_2=np.zeros(psnlty_usable.shape[1])
for dim_i in range(psnlty_usable.shape[1]):
	las_model.fit(cell_state,psnlty_usable[:,dim_i])
	psnlty_hat[:,dim_i]=las_model.predict(cell_state)
	R_2[dim_i]=las_model.score(cell_state,psnlty_usable[:,dim_i])


plt.scatter(psnlty_hat[:,4],psnlty_usable[:,4])
plt.plot([-3,3], [-3,3], color="black")
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.show()










