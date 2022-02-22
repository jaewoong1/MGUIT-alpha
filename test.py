import torch
import torchvision
from options import TestOptions
from dataset import testset
from model import MGUIT
import torch.nn.functional as F
import os
from tqdm import tqdm 

def main():
    # parse options
    parser = TestOptions()
    opts = parser.parse()
    torch.cuda.set_device(opts.gpu)

    # data loader
    print('\n--- load dataset ---')
    dataset = testset(opts)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=opts.nThreads)

    # model
    print('\n--- load model ---')
    model = MGUIT(opts)
    model.setgpu(opts.gpu)
    _, _ = model.resume(os.path.join(opts.result_dir, opts.name, opts.resume), opts.gpu, train=False)
    # saver for display and output

    # memory initialize
    print('\n---load memory items---')
    m_item = torch.load(os.path.join(opts.result_dir, opts.name, 'latest_memory.pt'))
    m_items = []
    for i in range(0, 4):
        tmp = m_item[i]
        m_item_ = []
        m_item_.append(tmp[0].cuda(opts.gpu))   # key
        m_item_.append(tmp[1].cuda(opts.gpu))   # value A
        m_item_.append(tmp[2].cuda(opts.gpu))   # value B
        m_items.append(m_item_)
    del m_item, tmp, m_item_
    
    image_dir = os.path.join(opts.save_dir, opts.name)
    # test
    print('\n--- test ---')
    if not os.path.exists(image_dir):
            os.makedirs(image_dir)

    for it, (images_a, images_b, masks_a, masks_b) in tqdm(enumerate(test_loader)):
        # input data
        images_a = images_a.cuda(opts.gpu).detach()
        images_b = images_b.cuda(opts.gpu).detach()
        masks_a = masks_a.cuda(opts.gpu).detach()
        masks_b = masks_b.cuda(opts.gpu).detach()

        images = model.generation(images_a, images_b, masks_a, masks_b, m_items)
        img_filename = '%s/generated_%05d.jpg' % (image_dir, it)
        torchvision.utils.save_image(images / 2 + 0.5, img_filename, nrow=1)
        
    return

if __name__ == '__main__':
    main()