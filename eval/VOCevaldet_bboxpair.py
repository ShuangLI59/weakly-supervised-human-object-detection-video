import pdb
import numpy as np
from eval.eval_utils import *

def VOCevaldet_bboxpair(det_id, det_bb, det_conf, gt_bbox, min_overlap, eval_criteria):
	npos = 0
	gt = {}

	for i in range(len(gt_bbox)):
		gt[i] = {}
		if len(gt_bbox[i])>0:
			gt[i]['bb'] = gt_bbox[i]
			gt[i]['det'] = np.zeros([gt_bbox[i].shape[0]])
			npos = npos + gt_bbox[i].shape[0]
		else:
			gt[i]['bb'] = []
			gt[i]['det'] = []

	# % sort detections by decreasing confidence
	si = np.argsort(det_conf)[::-1]
	
	if len(si)!=0:
		det_id = det_id[si]
		det_bb = det_bb[si, :]
	else:
		print('empty prediction')

	# % assign detections to ground truth objects
	nd = len(det_conf)
	tp = np.zeros([nd])
	fp = np.zeros([nd])

	for d in range(nd):
		i = det_id[d]
		bb_1 = det_bb[d, :4]
		bb_2 = det_bb[d, 4:]

		ov_max = -np.inf
		for j in range(len(gt[int(i)]['bb'])):
			# % get gt box
			bbgt_1 = gt[int(i)]['bb'][j, :4]
			bbgt_2 = gt[int(i)]['bb'][j, 4:]
			
			ov = compute_overlap(bb_1, bb_2, bbgt_1, bbgt_2, eval_criteria)
			if ov > ov_max:
				ov_max = ov
				j_max = j

		# % assign detection as true positive/don't care/false positive
		if ov_max >= min_overlap:
			if not gt[int(i)]['det'][j_max]:
				tp[d] = 1
				gt[int(i)]['det'][j_max] = True
			else:
				fp[d] = 1
		else:
			fp[d] = 1

	# % compute precision/recall
	fp = np.cumsum(fp)
	tp = np.cumsum(tp)
	rec = tp/npos
	prec = tp/(fp+tp)

	# % compute average precision
	ap = 0
	for t in np.arange(0,1,0.1):
		if len(prec[rec >= t])==0:
			p = 0
		else:
			p = np.max(prec[rec >= t])
		ap = ap + p/11
	
	return rec, prec, ap






