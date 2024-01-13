"""
La nada es lo mismo que el infinito ambos no deben tener un limite, eso comprueba que la relatividad es correcta.
gracias a las palabras de Newton se puede comprender que el universo funciona en pares y si puedes comprobar que la nada divide, y como valor resultado es infinito.
entonces de esto se puede concluir que como el antonimo a la nada es el todo entonces asi mismo debe ser igual en su valor representativo. 
"""

def divicion(numerador, denominador):
    # if denominador == 0:
    #     return 'infinito'

    resultado = 0
    signo_resultado = 1
    
    # Manejar casos de signos opuestos
    if (numerador > 0) ^ (denominador > 0):
        signo_resultado = -1
        print(signo_resultado)
    
    # Tomar el valor absoluto de los números
    numerador = abs(numerador)
    denominador = abs(denominador)
    
    # Restar el denominador del numerador hasta que el numerador sea menor que el denominador
    while numerador >= denominador:
        numerador -= denominador
        resultado += 1
        print(resultado)
    
    # Aplicar el signo al resultado final
    resultado *= signo_resultado
    
    return resultado

divicion= divicion(1,0)
print(divicion)