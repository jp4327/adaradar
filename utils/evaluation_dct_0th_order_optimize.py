import torch
import numpy as np
from .metrics import GetFullMetrics_withPerImage, Metrics
from .dct import snr, dct_based_compression, inject_noise
import pkbar

Sequences = {'Validation':['RECORD@2020-11-22_12.49.56','RECORD@2020-11-22_12.11.49','RECORD@2020-11-22_12.28.47','RECORD@2020-11-21_14.25.06'],
            'Test':['RECORD@2020-11-22_12.45.05','RECORD@2020-11-22_12.25.47','RECORD@2020-11-22_12.03.47','RECORD@2020-11-22_12.54.38']}

def run_evaluation(net,loader,encoder,check_perf=False, detection_loss=None,segmentation_loss=None,losses_params=None):

    metrics = Metrics()
    metrics.reset()

    net.eval()
    running_loss = 0.0
    
    kbar = pkbar.Kbar(target=len(loader), width=20, always_stateful=False)

    for i, data in enumerate(loader):

        # input, out_label,segmap,labels
        inputs = data[0].to('cuda').float()
        label_map = data[1].to('cuda').float()
        seg_map_label = data[2].to('cuda').double()

        with torch.set_grad_enabled(False):
            outputs = net(inputs)

        if(detection_loss!=None and segmentation_loss!=None):
            classif_loss,reg_loss = detection_loss(outputs['Detection'], label_map,losses_params)           
            prediction = outputs['Segmentation'].contiguous().flatten()
            label = seg_map_label.contiguous().flatten()        
            loss_seg = segmentation_loss(prediction, label)
            loss_seg *= inputs.size(0)
            
            classif_loss *= losses_params['weight'][0]
            reg_loss *= losses_params['weight'][1]
            loss_seg *=losses_params['weight'][2]


            loss = classif_loss + reg_loss + loss_seg

            # statistics
            running_loss += loss.item() * inputs.size(0)

        if(check_perf):
            out_obj = outputs['Detection'].detach().cpu().numpy().copy()
            labels = data[3]

            out_seg = torch.sigmoid(outputs['Segmentation']).detach().cpu().numpy().copy()
            label_freespace = seg_map_label.detach().cpu().numpy().copy()

            for pred_obj,pred_map,true_obj,true_map in zip(out_obj,out_seg,labels,label_freespace):

                metrics.update(pred_map[0],true_map,np.asarray(encoder.decode(pred_obj,0.05)),true_obj,
                            threshold=0.2,range_min=5,range_max=100) 
                
        kbar.update(i)
        

    mAP,mAR, mIoU = metrics.GetMetrics()

    return {'loss':running_loss, 'mAP':mAP, 'mAR':mAR, 'mIoU':mIoU}


def run_FullEvaluation_SGD(net,loader,encoder,args,config,
    iou_threshold=0.5,
    verify_quantize=False,
    quantize=False,
    result_only=False,
    ):

    net.eval()

    comp_ratio = args.comp_ratio # init comp ratio
    init_cr_per_scene = args.init_cr_per_scene
    seq_id_old = None
    ood_args = getattr(args, "OOD", None)

    kbar = pkbar.Kbar(target=len(loader), width=20, always_stateful=False)

    print('Generating Predictions...')
    predictions = {'prediction_p':{'objects':[],'freespace':[]},
                   'label_p':{'objects':[],'freespace':[]},
                   'prediction_m':{'objects':[],'freespace':[]},
                   'label_m':{'objects':[],'freespace':[]},
    }
    dct_info = {'SNR_p':[],
                'SNR_m':[],
                'cr_p':[],
                'cr_m':[],
                'grad_approx':[],
    }

    for i, data in enumerate(loader):

        inputs = data[0].float().detach().numpy()

        if ood_args:
            if args.period == 0:
                run_snr = args.snr
            elif args.ood_type == 'rect':
                run_snr = args.snr if i % args.period == 0 else 0 # create a rectangular pattern
            
            if run_snr != 0:
                inputs = inject_noise(inputs, run_snr)

        seq_id = data[5][0]
        if init_cr_per_scene:
            if seq_id != seq_id_old:
                comp_ratio = args.comp_ratio # init comp ratio
        seq_id_old = seq_id

        # 1. Perturb Parameter symmetrically
        comp_ratio_p = comp_ratio + args.epsilon # plus
        comp_ratio_m = comp_ratio - args.epsilon # minus

        # Enforce bounds on perturbed values *before* evaluation
        # This ensures we evaluate within the allowed parameter range
        comp_ratio_p_clipped = np.clip(comp_ratio_p, args.min_comp_ratio, args.max_comp_ratio)
        comp_ratio_m_clipped = np.clip(comp_ratio_m, args.min_comp_ratio, args.max_comp_ratio)

        # Apply DCT, thresholding, quantization, IDCT
        inputs_comp_p, dct_coef_p = dct_based_compression(inputs, comp_ratio_p_clipped, args.BL, quantize, args.qbit, verify_quantize)
        inputs_comp_m, dct_coef_m = dct_based_compression(inputs, comp_ratio_m_clipped, args.BL, quantize, args.qbit, verify_quantize)

        nonzero_count_p = np.count_nonzero(dct_coef_p) # count non-zero coefficients
        cr_p = inputs.size / nonzero_count_p if nonzero_count_p > 0 else float('inf')
        nonzero_count_m = np.count_nonzero(dct_coef_m) # count non-zero coefficients
        cr_m = inputs.size / nonzero_count_m if nonzero_count_m > 0 else float('inf')

        dct_info['cr_p'].append(cr_p)
        dct_info['cr_m'].append(cr_m)

        inputs_comp_p = torch.from_numpy(inputs_comp_p).to('cuda') #.float()
        inputs_comp_m = torch.from_numpy(inputs_comp_m).to('cuda') #.float()

        with torch.set_grad_enabled(False):
            out_p = net(inputs_comp_p)
            out_m = net(inputs_comp_m)

        out_obj_p = out_p['Detection'].detach().cpu().numpy().copy()
        out_seg_p = torch.sigmoid(out_p['Segmentation']).detach().cpu().numpy().copy()
        out_obj_m = out_m['Detection'].detach().cpu().numpy().copy()
        out_seg_m = torch.sigmoid(out_m['Segmentation']).detach().cpu().numpy().copy()
        
        labels_object = data[3]
        label_freespace = data[2].numpy().copy()

        for pred_obj,pred_map,true_obj,true_map in zip(out_obj_p,out_seg_p,labels_object,label_freespace):
            
            predictions['prediction_p']['objects'].append( np.asarray(encoder.decode(pred_obj,0.05)))
            predictions['label_p']['objects'].append(true_obj)

            predictions['prediction_p']['freespace'].append(pred_map[0])
            predictions['label_p']['freespace'].append(true_map)

        for pred_obj,pred_map,true_obj,true_map in zip(out_obj_m,out_seg_m,labels_object,label_freespace):
            
            predictions['prediction_m']['objects'].append( np.asarray(encoder.decode(pred_obj,0.05)))
            predictions['label_m']['objects'].append(true_obj)

            predictions['prediction_m']['freespace'].append(pred_map[0])
            predictions['label_m']['freespace'].append(true_map)

        # 2. Estimate Gradient
        if predictions['prediction_p']['objects'][-1].size > 0 and predictions['prediction_m']['objects'][-1].size > 0:
            loss_plus = predictions['prediction_p']['objects'][-1][:,2].max()
            loss_minus = predictions['prediction_m']['objects'][-1][:,2].max()
            denominator = comp_ratio_p_clipped - comp_ratio_m_clipped

            if abs(denominator) < 1/args.BL**2:
                grad_approx = 0.0 # Gradient is zero if no change in parameter value
            elif predictions['prediction_p']['objects'][-1][:,2].max() < args.conf_thd and predictions['prediction_m']['objects'][-1][:,2].max() < args.conf_thd:
                grad_approx = 0.0 # Gradient is zero if no change in parameter value
            else:
                grad_approx = (loss_plus - loss_minus) / denominator

            dct_info['grad_approx'].append(grad_approx)

            if args.loss_type == 'balance':
                denom1 = config['c_max'] - config['c_min']
                denom2 = config['r_max'] - config['r_min']
                r = (comp_ratio_p_clipped + comp_ratio_m_clipped) / 2
                c = (loss_plus + loss_minus) / 2
                lambda_ = args.lambda_val

                if denom1 == 0 or denom2 == 0:
                    # Handle the error appropriately, e.g., raise an exception or assign NaN
                    grad_approx = 0 # Example: Assign Not a Number
                else:
                    if args.objective == 'add':
                        grad_approx = grad_approx + lambda_ * (1/r**2)                                      # un-normalized
                    elif args.objective == 'norm':
                        grad_approx = (c - config['c_min']) + grad_approx*(r - config['r_min'] + lambda_)   # normalized

            grad_approx = np.clip(grad_approx, -args.grad_clip, args.grad_clip)

            # 3. Update Parameter (Gradient Descent Step)
            if args.enable_feedback:
                comp_ratio = comp_ratio + args.lr * grad_approx 

        # 4. Clip parameter to enforce bounds
        comp_ratio = np.clip(comp_ratio, args.min_comp_ratio, args.max_comp_ratio)
        kbar.update(i)

    results = GetFullMetrics_withPerImage(predictions['prediction_p']['objects'],
                                        predictions['label_p']['objects'],
                                        range_min=5,
                                        range_max=100,
                                        IOU_threshold=0.5)
    perfs, RangeError, AngleError, pi_prec_dict, pi_rec_dict = results
    mAP_per_scene = np.array([pi_prec_dict[threshold] for threshold in np.arange(0.1,0.96,0.1)])
    mAR_per_scene = np.array([pi_rec_dict[threshold] for threshold in np.arange(0.1,0.96,0.1)])
    
    mIoU = []
    for i in range(len(predictions['prediction_p']['freespace'])):
        # 0 to 124 means 0 to 50m
        pred = predictions['prediction_p']['freespace'][i][:124].reshape(-1)>=0.5
        label = predictions['label_p']['freespace'][i][:124].reshape(-1)
        
        intersection = np.abs(pred*label).sum()
        union = np.sum(label) + np.sum(pred) -intersection
        iou = intersection /union
        mIoU.append(iou)


    mIoU = np.asarray(mIoU).mean()
    print('------- Freespace Scores ------------')
    print('  mIoU', mIoU*100,'%')

    perfs_all = {'perfs': perfs,
                 'RangeError': RangeError,
                 'AngleError': AngleError,
                 'mIoU': mIoU,
                 'mAP_per_scene': mAP_per_scene,
                 'mAR_per_scene': mAR_per_scene,
    }

    
    if result_only:
        precision = perfs['precision']
        recall = perfs['recall']
        F1_score = (np.mean(precision)*np.mean(recall))/((np.mean(precision) + np.mean(recall))/2)
        result_dict = {'dct_info': dct_info,
                       'precision': np.mean(precision),
                       'recall': np.mean(recall),
                       'F1': F1_score,
                       'RangeError': np.mean(RangeError),
                       'AngleError': np.mean(AngleError),
                       'mIoU': mIoU,
                       'mAP_per_scene': mAP_per_scene,
                       'mAR_per_scene': mAR_per_scene,
                       }
        return result_dict
    else:
        return predictions, dct_info, perfs_all
