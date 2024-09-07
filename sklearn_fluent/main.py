def req(xlist, ylist, linearreg):
    from sklearn.linear_model import LinearRegression
    import numpy as np

    if linearreg == True:
        if len(ylist) > 50:
            from sklearn.model_selection import train_test_split
            x_train, x_test, y_train, y_test = train_test_split(np.array(xlist).reshape(-1, 1), np.array(ylist).reshape(-1, 1), test_size=0.2)
            model = LinearRegression()
            model.fit(x_train, y_train)
            accuracy = round(model.score(x_test, y_test))

        x_train = np.array(xlist).reshape(-1, 1)
        y_train = np.array(ylist).reshape(-1, 1)
        model = LinearRegression()
        model.fit(x_train, y_train)
    elif linearreg == False:
        x_train = np.array(xlist)
        y_train = np.array(ylist)
        model = LinearRegression()
        model.fit(x_train, y_train)

    a = model.intercept_
    b = model.coef_
    letters = list('abcdefghijklmnopqrstuvwxyz')
    reqletters = []
    for i in range(0, len(b)):
        reqletters.append(letters[i])
    newvars = []
    for i in range(len(reqletters)):
        try:
            new = str(round(b[0][i], 4)) + reqletters[i]  # Extract single element
        except:
            new = str(round(float(b[0][0]), 4)) + reqletters[i]  # Extract single element
        newvars.append(new)
    try:
        mainvar = round(a[0], 4)  # Extract single element
    except:
        mainvar = round(float(a[0]), 4)
    newvars.append(mainvar)
    last = " + ".join(list(map(str, newvars)))

    try:
        return f"function: {last}\naccuracy: {accuracy * 100}%"
    except:
        return f"function: {last}"
