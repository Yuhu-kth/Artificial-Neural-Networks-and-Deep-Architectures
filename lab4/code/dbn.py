from util import *
from rbm import RestrictedBoltzmannMachine

class DeepBeliefNet():    

    ''' 
    For more details : Hinton, Osindero, Teh (2006). A fast learning algorithm for deep belief nets. https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

    network          : [top] <---> [pen] ---> [hid] ---> [vis] 
                               `-> [lbl] 
    lbl : label
    top : top
    pen : penultimate
    hid : hidden
    vis : visible
    '''
    
    def __init__(self, sizes, image_size, n_labels, batch_size):

        """
        Args:
          sizes: Dictionary of layer names and dimensions
          image_size: Image dimension of data
          n_labels: Number of label categories
          batch_size: Size of mini-batch
        """

        self.rbm_stack = {
            
            'vis--hid' : RestrictedBoltzmannMachine(ndim_visible=sizes["vis"], ndim_hidden=sizes["hid"],
                                                    is_bottom=True, image_size=image_size, batch_size=batch_size),
            
            'hid--pen' : RestrictedBoltzmannMachine(ndim_visible=sizes["hid"], ndim_hidden=sizes["pen"], batch_size=batch_size),
            
            'pen+lbl--top' : RestrictedBoltzmannMachine(ndim_visible=sizes["pen"]+sizes["lbl"], ndim_hidden=sizes["top"],
                                                        is_top=True, n_labels=n_labels, batch_size=batch_size)
        }
        
        self.sizes = sizes

        self.image_size = image_size

        self.batch_size = batch_size
        
        self.n_gibbs_recog = 5
        
        self.n_gibbs_gener = 200
        
        self.n_gibbs_wakesleep = 5

        self.print_period = 2000
        
        return

    def recognize(self,true_img,true_lbl):

        """Recognize/Classify the data into label categories and calculate the accuracy

        Args:
          true_imgs: visible data shaped (number of samples, size of visible layer)
          true_lbl: true labels shaped (number of samples, size of label layer). Used only for calculating accuracy, not driving the net
        """
        
        n_samples = true_img.shape[0]
        
        vis = true_img # visible layer gets the image data
        
        lbl = np.ones(true_lbl.shape)/10. # start the net by telling you know nothing about labels        
        
        # [TODO TASK 4.2] fix the image data in the visible layer and drive the network bottom to top. In the top RBM, run alternating Gibbs sampling \
        # and read out the labels (replace pass below and 'predicted_lbl' to your predicted labels).
        # NOTE : inferring entire train/test set may require too much compute memory (depends on your system). In that case, divide into mini-batches.
        
        _,hidden = self.rbm_stack["vis--hid"].get_h_given_v_dir(vis)
        _,Pen = self.rbm_stack["hid--pen"].get_h_given_v_dir(hidden)

        topVis = np.hstack((Pen, lbl))
        #print(topVis.shape)
        #topVis[:,:,-true_lbl[1]] = Pen

        for _ in range(self.n_gibbs_recog):
            _, topH = self.rbm_stack["pen+lbl--top"].get_h_given_v(topVis)
            topVis,_=self.rbm_stack["pen+lbl--top"].get_v_given_h(topH)

        predicted_lbl = topVis[:,-true_lbl.shape[1]:]
            
        print ("accuracy = %.2f%%"%(100.*np.mean(np.argmax(predicted_lbl,axis=1)==np.argmax(true_lbl,axis=1))))
        
        return

    def generate(self,true_lbl,name):
        
        """Generate data from labels

        Args:
          true_lbl: true labels shaped (number of samples, size of label layer)
          name: string used for saving a video of generated visible activations
        """
        
        n_sample = true_lbl.shape[0]
        
        records = []        
        fig,ax = plt.subplots(1,1,figsize=(3,3))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax.set_xticks([]); ax.set_yticks([])

        lbl = true_lbl

        # [TODO TASK 4.2] fix the label in the label layer and run alternating Gibbs sampling in the top RBM. From the top RBM, drive the network \ 
        # top to the bottom visible layer (replace 'vis' from random to your generated visible layer).
        top = self.rbm_stack['pen+lbl--top']
        pen = self.rbm_stack['hid--pen']
        hid = self.rbm_stack['vis--hid']

        prob_topVis = np.random.uniform(0,1,(lbl.shape[0], top.bias_v.shape[0]))
        topVis = prob_topVis
        #topVis = np.random.binomial(1,prob_topVis)


        #print(prob_topVis.shape)
        #topVis = np.random.binomial(1, np.abs(prob_topVis.reshape(1,-1)))

        #print(topVis)


        for j in range(self.n_gibbs_gener):
            topVis[:,-lbl.shape[1]:] = lbl
            _,topH = top.get_h_given_v(topVis)
            topVis,_ = top.get_v_given_h(topH)
            #print(topVis.shape)
            #print(topVis[0,:-lbl.shape[1]].shape)
            #print(lbl.shape)

            _,visPen = pen.get_v_given_h_dir(topVis[:,:-lbl.shape[1]])
            vis,_ = hid.get_v_given_h_dir(visPen)



            records.append( [ ax.imshow(vis.reshape(self.image_size), cmap="bwr", vmin = 0, vmax = 1, animated=True, interpolation=None) ] )
            """if j == self.n_gibbs_gener-1:  
                plt.imshow(vis.reshape(self.image_size), cmap = 'bwr')
                plt.savefig("%d.png"%np.argmax(true_lbl))"""

        anim = stitch_video(fig,records).save("%s.generate%d.mp4"%(name,np.argmax(true_lbl)))            
            
        return

    def train_greedylayerwise(self, vis_trainset, lbl_trainset, n_iterations):

        """
        Greedy layer-wise training by stacking RBMs. This method first tries to load previous saved parameters of the entire RBM stack. 
        If not found, learns layer-by-layer (which needs to be completed) .
        Notice that once you stack more layers on top of a RBM, the weights are permanently untwined.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        try :

            self.loadfromfile_rbm(loc="trained_rbm",name="vis--hid")
            self.rbm_stack["vis--hid"].untwine_weights()            
            
            self.loadfromfile_rbm(loc="trained_rbm",name="hid--pen")
            self.rbm_stack["hid--pen"].untwine_weights()
            
            self.loadfromfile_rbm(loc="trained_rbm",name="pen+lbl--top")        

        except IOError :

            # [TODO TASK 4.2] use CD-1 to train all RBMs greedily
        
            print ("training vis--hid")
            """ 
            CD-1 training for vis--hid 
            """   
            currentRBM = self.rbm_stack["vis--hid"]
            currentRBM.cd1(vis_trainset, n_iterations)         
            self.savetofile_rbm(loc="trained_rbm",name="vis--hid")
            _,hidden = currentRBM.get_h_given_v(vis_trainset)

            print ("training hid--pen")
            self.rbm_stack["vis--hid"].untwine_weights()            
            """ 
            CD-1 training for hid--pen 
            """            
            currentRBM = self.rbm_stack["hid--pen"]
            currentRBM.cd1(hidden, n_iterations)
            self.savetofile_rbm(loc="trained_rbm",name="hid--pen")   
            _,Pen = currentRBM.get_h_given_v(hidden)         

            print ("training pen+lbl--top")
            self.rbm_stack["hid--pen"].untwine_weights()
            """ 
            CD-1 training for pen+lbl--top 
            """
            pen_lbl = np.hstack((Pen, lbl_trainset))
            currentRBM = self.rbm_stack["pen+lbl--top"]
            currentRBM.cd1(pen_lbl,n_iterations)
            self.savetofile_rbm(loc="trained_rbm",name="pen+lbl--top")            

        return    

    def train_wakesleep_finetune(self, vis_trainset, lbl_trainset, n_iterations):

        """
        Wake-sleep method for learning all the parameters of network. 
        First tries to load previous saved parameters of the entire network.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """
        
        print ("\ntraining wake-sleep..")

        try :
            
            self.loadfromfile_dbn(loc="trained_dbn",name="vis--hid")
            self.loadfromfile_dbn(loc="trained_dbn",name="hid--pen")
            self.loadfromfile_rbm(loc="trained_dbn",name="pen+lbl--top")
            
        except IOError :            

            self.n_samples = vis_trainset.shape[0]


            visHid = self.rbm_stack["vis--hid"]
            hidPen = self.rbm_stack["hid--pen"]
            penLblTop = self.rbm_stack["pen+lbl--top"]

            for it in range(n_iterations): 
                ind = np.random.uniform(size = self.batch_size, high = self.n_samples).astype(int)
                #ind2 = np.random.uniform(size = self.batch_size, high = self.n_samples).astype(int)

                vis_minibatch = vis_trainset[ind]
                lbl_minibatch = lbl_trainset[ind]           
                                                
                # [TODO TASK 4.3] wake-phase : drive the network bottom to top using fixing the visible and label data.
                _,wakehidstates = visHid.get_h_given_v_dir(vis_minibatch)
                _,wakepenstates = hidPen.get_h_given_v_dir(wakehidstates)
                _,waketopstates = penLblTop.get_h_given_v(np.hstack((wakepenstates, lbl_minibatch)))
                negtoptates = waketopstates
                #print(tV.shape)
                #_,topH = penLblTop.get_h_given_v(topV)
                # [TODO TASK 4.3] alternating Gibbs sampling in the top RBM for k='n_gibbs_wakesleep' steps, also store neccessary information for learning this RBM.
                for _ in range(self.n_gibbs_wakesleep):
                    negpen,negpenstates = penLblTop.get_v_given_h(negtoptates)
                    negtop,negtoptates = penLblTop.get_h_given_v(negpen)

                # [TODO TASK 4.3] sleep phase : from the activities in the top RBM, drive the network top to bottom.
                #print(topH.shape)
                sleeppenstates = negpenstates
                _,sleephidstates =hidPen.get_v_given_h_dir(sleeppenstates[:,:-lbl_minibatch.shape[1]])
                sleepvisprob,_ = visHid.get_v_given_h_dir(sleephidstates)
                # [TODO TASK 4.3] compute predictions : compute generative predictions from wake-phase activations, and recognize predictions from sleep-phase activations.
                # Note that these predictions will not alter the network activations, we use them only to learn the directed connections.
                #rec
                psleeppenstates,_ = hidPen.get_h_given_v_dir(sleephidstates)
                psleephidstates,_ = visHid.get_h_given_v_dir(sleepvisprob)
                #gen
                pvisprobs,_ = visHid.get_v_given_h_dir(wakehidstates)
                phidprobs,_ = hidPen.get_v_given_h_dir(wakepenstates)

                # [TODO TASK 4.3] update generative parameters : here you will only use 'update_generate_params' method from rbm class.
                visHid.update_generate_params(wakehidstates, vis_minibatch, pvisprobs)
                hidPen.update_generate_params(wakepenstates, wakehidstates, phidprobs)
                # [TODO TASK 4.3] update parameters of top rbm : here you will only use 'update_params' method from rbm class.
                #if it%500 == 0:
                #    print(np.sum(np.square(negpenstates - np.hstack((wakepenstates,lbl_minibatch)))))
                penLblTop.update_params(np.hstack((wakepenstates,lbl_minibatch)), waketopstates, negpenstates, negtoptates)
                # [TODO TASK 4.3] update generative parameters : here you will only use 'update_recognize_params' method from rbm class.
                hidPen.update_recognize_params(sleephidstates, sleeppenstates[:,:-lbl_minibatch.shape[1]], psleeppenstates)
                visHid.update_recognize_params(sleepvisprob, sleephidstates, psleephidstates)
                if it % self.print_period == 0 : print ("iteration=%7d"%it)
                        
            self.savetofile_dbn(loc="trained_dbn",name="vis--hid")
            self.savetofile_dbn(loc="trained_dbn",name="hid--pen")
            self.savetofile_rbm(loc="trained_dbn",name="pen+lbl--top")            

        return

    
    def loadfromfile_rbm(self,loc,name):
        
        self.rbm_stack[name].weight_vh = np.load("%s/rbm.%s.weight_vh.npy"%(loc,name))
        self.rbm_stack[name].bias_v    = np.load("%s/rbm.%s.bias_v.npy"%(loc,name))
        self.rbm_stack[name].bias_h    = np.load("%s/rbm.%s.bias_h.npy"%(loc,name))
        print ("loaded rbm[%s] from %s"%(name,loc))
        return
        
    def savetofile_rbm(self,loc,name):
        
        np.save("%s/rbm.%s.weight_vh"%(loc,name), self.rbm_stack[name].weight_vh)
        np.save("%s/rbm.%s.bias_v"%(loc,name),    self.rbm_stack[name].bias_v)
        np.save("%s/rbm.%s.bias_h"%(loc,name),    self.rbm_stack[name].bias_h)
        return
    
    def loadfromfile_dbn(self,loc,name):
        
        self.rbm_stack[name].weight_v_to_h = np.load("%s/dbn.%s.weight_v_to_h.npy"%(loc,name))
        self.rbm_stack[name].weight_h_to_v = np.load("%s/dbn.%s.weight_h_to_v.npy"%(loc,name))
        self.rbm_stack[name].bias_v        = np.load("%s/dbn.%s.bias_v.npy"%(loc,name))
        self.rbm_stack[name].bias_h        = np.load("%s/dbn.%s.bias_h.npy"%(loc,name))
        print ("loaded rbm[%s] from %s"%(name,loc))
        return
        
    def savetofile_dbn(self,loc,name):
        
        np.save("%s/dbn.%s.weight_v_to_h"%(loc,name), self.rbm_stack[name].weight_v_to_h)
        np.save("%s/dbn.%s.weight_h_to_v"%(loc,name), self.rbm_stack[name].weight_h_to_v)
        np.save("%s/dbn.%s.bias_v"%(loc,name),        self.rbm_stack[name].bias_v)
        np.save("%s/dbn.%s.bias_h"%(loc,name),        self.rbm_stack[name].bias_h)
        return
    
