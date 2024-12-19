import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sla

# ## Problema
# 4 utenti e loro valutazioni per 6 stagioni
# 
# |        | Ryne | Erin | Nathan | Pete |
# | --- | --- | --- | --- | --- |
# | season 1 | 5 | 5 | 0 | 5 |
# | season 2 | 5 | 0 | 3 | 4 |
# | season 3 | 3 | 4 | 0 | 3 |
# | season 4 | 0 | 0 | 5 | 3 |
# | season 5 | 5 | 4 | 4 | 5 |
# | season 6 | 5 | 4 | 5 | 5 |


for k in [2, 3, 4]:
    print(f'k={k}')
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

    # ## Calcoliamo la SVD di A

    print('dimensions of U, s and V')
    U, S, V = sla.svd(A)
    print(U.shape)
    print(S.shape)
    print(V.shape)
    print('vector of singular values=',S)


    # ### rappresentiamo i dati in uno spazio a due dimensioni

    U2 = U[:,:k]
    V2 = V.T[:,:k]
    S2 = np.diag(S[:k])
    print(f'SVD troncata con k={k}. Stampa di U, s e V con 2 cifre decimali')
    print('U2=',U2.round(2))
    print('S2=',S2.round(2))
    print('V2=',V2.round(2))


    # ### Grafichiamo la proiezione dei dati in uno spazio 2D
    # 
    # Le prime due colonne di U rappresentano le stagioni. Le prime due righe di V rappresentano gli utenti.
    # 
    #  x rappresenti la prima componente, y la seconda


    plt.plot(U2[:,0], U2[:,1], 'bo', markersize=15, clip_on=False, label='seasons')
    plt.plot(V2[:,0], V2[:,1], 'rs', markersize=15, clip_on=False, label='users')

    ax = plt.gca()
    for i, txt in enumerate(seasons):
        ax.text(U2[i,0], U2[i,1], txt, ha='left', va='bottom', fontsize=20)
        
    for i, txt in enumerate(users):
        ax.text(V2[i,0], V2[i,1], txt, ha='left', va='bottom', fontsize=20)

    # axis trickery
    ax = plt.gca()
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_position('center')
    ax.spines['top'].set_color('none')
    ax.spines['left']
    ax.spines['bottom']
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('right')


    # ## Obiettivo: trovare utenti simili per fornire raccomandazioni
    # 
    # Aggiungiamo l'utente Luke, che dà le valutazioni seguenti per le stagioni:
    # 5,5,0,0,0,5.
    # Per fornire raccomandazioni a Luke troviamo gli utenti più simili a Luke,
    # rappresentando Luke nello spazio 2D in cui abbiamo rappresentato gli utenti

    luke = np.array([5,5,0,0,0,5])
    print('valutazioni delle stagioni di Luke',luke)


    # 
    # Per proiettare le valutazioni di Luke nello spazio 2D degli utenti, se esse sono indicate con L
    # 
    # L^T * U_2 * S_2^{-1}
    # 

    luke2d = luke.dot(U2.dot(np.linalg.inv(S2)))
    print('valutazioni di Luke proiettate nello spazio 2D')
    print(luke2d)


    # 
    # Grafichiamo le valutazioni di Luke così rappresentate

    plt.plot(U2[:,0], U2[:,1], 'bo', markersize=15, clip_on=False, label='seasons')
    plt.plot(V2[:,0], V2[:,1], 'rs', markersize=15, clip_on=False, label='users')

    ax = plt.gca()
    for i, txt in enumerate(seasons):
        ax.text(U2[i,0], U2[i,1], txt, ha='left', va='bottom', fontsize=20)
        
    for i, txt in enumerate(users):
        ax.text(V2[i,0], V2[i,1], txt, ha='left', va='bottom', fontsize=20)
        
    plt.plot(luke2d[0], luke2d[1], 'g*', markersize=15, clip_on=False, label='luke')
    ax.text(luke2d[0], luke2d[1], 'Luke', ha='left', va='bottom', fontsize=20)


    # axis trickery
    ax = plt.gca()
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_position('center')
    ax.spines['top'].set_color('none')
    ax.spines['left']
    ax.spines['bottom']
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('right')

    print(luke2d.shape)

    most_similar_users = []

    for i,xy in enumerate(V2):
        angle = np.dot(xy, luke2d) / (np.linalg.norm(xy) * np.linalg.norm(luke2d))
        print("coseno dell'angolo tra %s e %s: %2.2g" % ('luke', users[i], angle))
        most_similar_users.append((users[i], float(round(angle, 3))))

    most_similar_users.sort(key=lambda x: x[1], reverse=True)

    plt.savefig(f'SVD_rs-k{k}.png')

    print()

    print('utenti più simili a Luke')
    for i, (key, value) in enumerate(most_similar_users):
        print("\t", i + 1, key, value)
    print()
    print("=" * 100)