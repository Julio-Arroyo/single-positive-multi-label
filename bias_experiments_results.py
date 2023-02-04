import matplotlib.pyplot as plt
import numpy as np


LOC = 2
SIZE = 1
EMPIRICAL = 3
UNIFORM = 4


coco_em_location = [63.63802065671423, 63.55851762988382, 63.63072831673501]
coco_em_size = [58.15687938368027, 58.102806719508195, 58.003629325450156]
coco_em_empir = [61.21261467649543, 62.05261316053063, 62.158175015682794]
coco_em_unif = [64.86, 64.95]
coco_em = coco_em_location + coco_em_size + coco_em_empir + coco_em_unif
coco_em_xaxis = 3*[LOC] + 3*[SIZE] + 3 *[EMPIRICAL] + 2*[UNIFORM]

coco_an_location = [61.14141833,
60.92943675,
60.75481325,
60.92968922,
60.86188918]
coco_an_size = [57.02210727
,56.90092269
,57.00530053
,56.99406775
,57.01016685]
coco_an_empir = [59.38177819,
60.27201343,
59.85162722,
59.92420769,
59.5107008]
coco_an_uniform = [62.3]
coco_an = coco_an_location[:3] + coco_an_size[:3] + coco_an_empir[:3] + coco_an_uniform
coco_an_xaxis = 3*[LOC] + 3*[SIZE] + 3*[EMPIRICAL] + [UNIFORM,]

coco_anls_loc = [62.85943022750905, 62.582046956167844, 62.678313689989054]
coco_anls_size = [57.1269298,
56.46388672,
56.434244]
coco_anls_uniform = [64.8]
coco_anls_empir = [59.60763148434063, 59.90604526092811, 59.88292131132186]
coco_anls = coco_anls_loc + coco_anls_size + coco_anls_empir + coco_anls_uniform
coco_anls_xaxis = 3*[LOC] + 3*[SIZE] + 3*[EMPIRICAL] + [UNIFORM,]

coco_role_loc = [66.37806875423662, 66.34334757830585, 66.42292317167528]
coco_role_size = [59.98912676, 60.22807157]
coco_role_empir = [66.4026448407274, 66.40165083821367, 66.37174394831656]
coco_role_uniform = [66.3]
coco_role = coco_role_loc + coco_role_size + coco_role_empir + coco_role_uniform
coco_role_xaxis = 3*[LOC] + 2*[SIZE] + 3*[EMPIRICAL] + [UNIFORM,]


# plot EM loss
plt.scatter(coco_em_xaxis, coco_em)

# plot AN loss
plt.scatter(coco_an_xaxis, coco_an)

# plot ANLS loss
plt.scatter(coco_anls_xaxis, coco_anls)

# plot ROLE loss
plt.scatter(coco_role_xaxis, coco_role)

plt.xlabel('Bias')
plt.xticks([SIZE, LOC, EMPIRICAL, UNIFORM], ['Size', 'Location', 'Empirical', 'Uniform'])
plt.ylabel('test MAP')
plt.yticks(np.arange(56, 67, step=1))
plt.title('COCO SPML Bias Experiments')
plt.legend(['EM', 'AN', 'ANLS', 'ROLE'])
plt.savefig('update_results.png')
