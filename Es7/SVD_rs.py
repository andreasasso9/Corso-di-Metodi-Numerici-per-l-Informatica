import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sla
from pathlib import Path

# Dati
seasons = [1,2,3,4,5,6]
users = ['Ryne', 'Erin', 'Nathan', 'Pete']

A = np.array([            
    [5,5,0,5], # season 1
    [5,0,3,4], # season 2
    [3,4,0,3], # season 3
    [0,0,5,3], # season 4
    [5,4,4,5], # season 5
    [5,4,5,5]  # season 6
    ], dtype=float)

# SVD completa
U, s, VT = sla.svd(A)
V = VT.T

luke = np.array([5,5,0,0,0,5])
print('Valutazioni di Luke:', luke)

# Funzione per SVD troncata e proiezione di Luke
def svd_projection(k):
    U_k = U[:,:k]
    V_k = V[:,:k]
    S_k = np.diag(s[:k])

    print(f'\n--- SVD troncata con k={k} ---')
    print('U_k=\n', U_k.round(2))
    print('S_k=\n', S_k.round(2))
    print('V_k=\n', V_k.round(2))

    # Proiezione di Luke
    luke_k = luke.dot(U_k.dot(np.linalg.inv(S_k)))
    print('Valutazioni di Luke proiettate nello spazio kD:', luke_k)

    # Similarità coseno
    for i, xy in enumerate(V_k):
        cos_angle = np.dot(xy, luke_k) / (np.linalg.norm(xy) * np.linalg.norm(luke_k))
        print(f"Coseno tra Luke e {users[i]}: {cos_angle:.2f}")
        if cos_angle > 0.9:
            print(f"  => {users[i]} è molto simile a Luke")

    # Grafico
    if k == 2:  # solo 2D
        plt.figure(figsize=(8,6))
        plt.plot(U_k[:,0], U_k[:,1], 'bo', markersize=15, label='seasons')
        plt.plot(V_k[:,0], V_k[:,1], 'rs', markersize=15, label='users')
        plt.plot(luke_k[0], luke_k[1], 'g*', markersize=15, label='Luke')

        ax = plt.gca()
        for i, txt in enumerate(seasons):
            ax.text(U_k[i,0], U_k[i,1], txt, ha='left', va='bottom', fontsize=12)
        for i, txt in enumerate(users):
            ax.text(V_k[i,0], V_k[i,1], txt, ha='left', va='bottom', fontsize=12)
        ax.text(luke_k[0], luke_k[1], 'Luke', ha='left', va='bottom', fontsize=12)

        # spines
        ax.spines['left'].set_color('none')
        ax.spines['bottom'].set_position('center')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('right')
        plt.legend()
        plt.title(f'Proiezione SVD k={k}')
        path = Path('Es7/img')
        path.mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{path}/svd_projection_k{k}.png')

# Eseguiamo per k=2,3,4
svd_projection(2)
svd_projection(3)
svd_projection(4)
