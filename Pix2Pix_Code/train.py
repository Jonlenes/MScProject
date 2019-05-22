import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from util.Timer import Timer
from util.util import tensor2im
from util.eval_metric import score


if __name__ == '__main__':
    
    # orig_stdout = sys.stdout
    # f = open('out.txt', 'w')
    # sys.stdout = f
    
    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_steps = 0
    timer = Timer()
    timer.start()

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        acumulate_acc = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            
            visualizer.reset()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            acumulate_acc += score(tensor2im(model.real_A), tensor2im(model.fake_B))
            
            """if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                save_suffix = 'iter_%d' % total_steps if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)"""
            
            print(i, score(tensor2im(model.real_A), tensor2im(model.fake_B)))
            iter_data_time = time.time()
        

        losses = model.get_current_losses()
        visualizer.print_current_losses(epoch, epoch_iter, losses, 0, 0)
        if opt.display_id > 0:
            visualizer.plot_current_losses(epoch, epoch_iter, opt, losses)
        print("Training time so far:", timer.diff())
        
        # salvando os resultados
        save_result = total_steps % opt.update_html_freq == 0
        visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save_networks('latest')
            # model.save_networks(epoch)
        
        print("Len:", len(dataset))
        print('End of epoch %d / %d \t Time Taken: %d sec \t Acc: %d' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time, int(acumulate_acc/len(dataset))))
        model.update_learning_rate()
        
    print("All traning time:", timer.diff())
    
    # sys.stdout = orig_stdout
    # f.close()

