import matplotlib
import matplotlib.pyplot as plt


class GANTrainer:
    def __init__(self, GANModel, dataset_real, dataset_generator, preview_epoch=True, 
                 save_epoch_interval=10, num_imgs_save=25, make_grid=True, save_path="./result/"):
        
        self.dataset_real = dataset_real
        self.dataset_generator = dataset_generator
        self.save_epoch_interval = save_epoch_interval
        self.num_imgs_save = num_imgs_save
        self.make_grid = make_grid
        self.save_path = save_path
        self.preview_epoch = preview_epoch
        self.GANModel = GANModel


    def fit(self, epochs=50, batch_size=32):
        
        print("Starting train for %d epochs" % epochs)
        timer = Timer()

        for epoch in range(1, epochs+1):
            timer.start()
            while self.dataset_real.has_next():
                #-----------------DISCRIMINATOR-----------------#
                # Dados do conjunto de treinamento real
                x_train_real = self.dataset_real.nexts(batch_size)
                y_train_real = np.ones((len(x_train_real), 1))

                # Dados fakes gerados pelo Gerador
                x_train_fake = self.GANModel.generator().predict( self.dataset_generator.nexts(batch_size) )
                y_train_fake = np.zeros((batch_size, 1))
                
                # Treinando por um batch
                d_loss_real = self.GANModel.discriminator().train_on_batch(x_train_real, y_train_real)
                d_loss_fake = self.GANModel.discriminator().train_on_batch(x_train_fake, y_train_fake)
                
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                #-----------------DISCRIMINATOR-----------------#


                #-------------------GENERATOR-------------------#
                x = self.dataset_generator.nexts(batch_size)
                y = np.ones((len(x), 1)) # Dizer que é real

                a_loss = self.GANModel.adversarial().train_on_batch(x, y)
                #-------------------GENERATOR-------------------#

            self.dataset_real.iteration_to_begin()
                
            # Model evolution
            print("Epoch %02d - Time %s  - [D loss: %.2f, acc.: %.2f%%] [G loss: %.2f]" % (epoch, timer.elapsed_time(), d_loss[0], 100 * d_loss[1], a_loss))

            # Saving sample
            self.show_save_samples(epoch, epoch % self.save_epoch_interval==0)

                        
    def show_save_samples(self, epoch, save):
        self._make_img_grid(epoch, save and self.make_grid)
        if save and not self.make_grid:
            self._save_genereted_image(epoch)
      
                            
    def _make_img_grid(self, epoch, save=False):

        r = c = int(np.sqrt(self.num_imgs_save))

        gen_imgs = self.GANModel.generator().predict( self.dataset_generator.nexts(r * c) )
        #Normalização pode ser necessária -  gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        count = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[count, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                count += 1

        if save:
            fig.savefig(self.save_path + "result_epoch_%d.png" % (epoch))
            print("Image grid result save.")

        if self.preview_epoch:
            plt.show()

        plt.close()