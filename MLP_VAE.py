class MLP_VAE:
    def __init__(self,input_dim,lat_dim, outliers_fraction):     
        self.outliers_fraction = outliers_fraction # for computing the threshold of anomaly score       
        self.input_dim = input_dim
        self.lat_dim = lat_dim # the lat_dim can exceed input_dim    
        
        self.input_X = tf.placeholder(tf.float32,shape=[None,self.input_dim],name='source_x')
        
        self.learning_rate = 0.0005
        self.batch_size =  32
        # batch_size should be smaller than normal setting for getting
        # a relatively lower anomaly-score-threshold
        self.train_iter = 3000
        self.hidden_units = 128
        self._build_VAE()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.pointer = 0
        
    def _encoder(self):
        with tf.variable_scope('encoder',reuse=tf.AUTO_REUSE):
            l1 = build_dense(self.input_X,self.hidden_units,activation=lrelu)

            l2 = build_dense(l1,self.hidden_units,activation=lrelu)
          
            mu = tf.layers.dense(l2,self.lat_dim)
            sigma = tf.layers.dense(l2,self.lat_dim,activation=tf.nn.softplus)
            sole_z = mu + sigma *  tf.random_normal(tf.shape(mu),0,1,dtype=tf.float32)
        return mu,sigma,sole_z
        
    def _decoder(self,z):
        with tf.variable_scope('decoder',reuse=tf.AUTO_REUSE):
            l1 = build_dense(z,self.hidden_units,activation=lrelu)

            l2 = build_dense(l1,self.hidden_units,activation=lrelu)

            recons_X = tf.layers.dense(l2,self.input_dim)        # 다시 원래 차원으로 복원
        return recons_X           # 복원된 x

    def _build_VAE(self):
        self.mu_z,self.sigma_z,sole_z = self._encoder()
        self.recons_X = self._decoder(sole_z)
        
        with tf.variable_scope('loss'):
            KL_divergence = 0.5 * tf.reduce_sum(tf.square(self.mu_z) + tf.square(self.sigma_z) - tf.log(1e-8 + tf.square(self.sigma_z)) - 1, 1)
            mse_loss = tf.reduce_sum(tf.square(self.input_X-self.recons_X), 1)          
            self.all_loss =  mse_loss  
            self.loss = tf.reduce_mean(mse_loss + KL_divergence)
            
        with tf.variable_scope('train'):            
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            

    def _fecth_data(self,input_data):        
        if (self.pointer+1) * self.batch_size  >= input_data.shape[0]:
            return_data = input_data[self.pointer*self.batch_size:,:]
            self.pointer = 0
        else:
            return_data =  input_data[ self.pointer*self.batch_size:(self.pointer+1)*self.batch_size,:]
            self.pointer = self.pointer + 1
        return return_data

    def train(self,train_X):
        for index in range(self.train_iter):
            this_X = self._fecth_data(train_X)
            self.sess.run([self.train_op],feed_dict={ self.input_X: this_X })
        self.arrage_recons_loss(train_X)

        
    def arrage_recons_loss(self,input_data):      # reconsturction error threshold 세팅
        all_losses =  self.sess.run(self.all_loss,feed_dict={self.input_X: input_data })
        self.test_loss = np.percentile(all_losses,(1-self.outliers_fraction)*100)         # train을 통해 reconst loss 설정
                

    def test(self,input_data):
        return_label = []
        print("Reconstruction threshold : ", self.test_loss)
        for index in range(input_data.shape[0]):    # test 길이인 8957
            single_X = input_data[index].reshape(1,-1)
            this_loss = self.sess.run(self.loss,feed_dict={ self.input_X: single_X })
            if this_loss < self.test_loss:
                return_label.append(0)        # 정상
            else:
                return_label.append(1)        # 비정상
        return return_label
 
def mlp_vae_predict(train,test,test_label):
    mlp_vae = MLP_VAE(30,20,0.1)
    mlp_vae.train(train)
    mlp_vae_predict_label = mlp_vae.test(test) 
    
    plot_confusion_matrix(test_label, mlp_vae_predict_label, ['Normal','Anomaly'],'MLP_VAE Confusion-Matrix')

if __name__ == '__main__':
    mlp_vae_predict(train, test, test_label)
