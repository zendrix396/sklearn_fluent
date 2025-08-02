def req(xlist, ylist):
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import train_test_split
    import numpy as np

    x_arr = np.array(xlist)
    y_arr = np.array(ylist)

    best_model = None
    best_score = -float('inf')
    best_degree = 0 # 0 for multi-linear, 1,2,3 for poly
    final_poly_feature_names = None
    
    # Determine if xlist is for a single feature or multiple features
    is_single_feature = False
    if x_arr.ndim == 1:
        is_single_feature = True
        x_processed_for_poly = x_arr.reshape(-1, 1)
    elif x_arr.ndim == 2 and x_arr.shape[1] == 1:
        is_single_feature = True
        x_processed_for_poly = x_arr
    else: # Multiple features
        x_processed_linear = x_arr

    if is_single_feature:
        possible_degrees = []
        if len(y_arr) >= 2: possible_degrees.append(1)
        if len(y_arr) >= 3: possible_degrees.append(2)
        if len(y_arr) >= 4: possible_degrees.append(3) # Max degree set to 3 for simplicity

        if not possible_degrees and len(y_arr) >=2 : # Should at least have degree 1 if enough points
            possible_degrees.append(1)
        elif not possible_degrees:
             return "Error: Not enough data points to fit a model."

        for degree in possible_degrees:
            current_pipeline = make_pipeline(
                PolynomialFeatures(degree=degree, include_bias=False),
                LinearRegression()
            )
            
            score_on_test = len(y_arr) > 50
            if score_on_test:
                x_train, x_test, y_train, y_test = train_test_split(
                    x_processed_for_poly, y_arr, test_size=0.2, random_state=42
                )
                current_pipeline.fit(x_train, y_train)
                current_score = current_pipeline.score(x_test, y_test)
                # Re-fit on full data for final model parameters if this is the best model
            else:
                current_pipeline.fit(x_processed_for_poly, y_arr)
                current_score = current_pipeline.score(x_processed_for_poly, y_arr)

            if current_score > best_score:
                best_score = current_score
                best_degree = degree
                # If scored on test, refit on full data to get final coefficients
                if score_on_test:
                    current_pipeline.fit(x_processed_for_poly, y_arr)
                best_model = current_pipeline
        
        if best_model:
            final_poly_feature_names = best_model.named_steps['polynomialfeatures'].get_feature_names_out()
            model_for_coeffs = best_model.named_steps['linearregression']
        else: # Fallback if no model was chosen (should not happen if possible_degrees is populated)
            return "Error: Could not determine a suitable model."

    else: # Multiple Linear Regression
        best_degree = 0 # Mark as multi-linear
        model_for_coeffs = LinearRegression()
        score_on_test = len(y_arr) > 50
        if score_on_test:
            x_train, x_test, y_train, y_test = train_test_split(
                x_processed_linear, y_arr, test_size=0.2, random_state=42
            )
            # Fit a separate model for scoring, then fit the main model on all data
            scoring_model = LinearRegression()
            scoring_model.fit(x_train, y_train)
            best_score = scoring_model.score(x_test, y_test)
            model_for_coeffs.fit(x_processed_linear, y_arr) # Fit on full data
        else:
            model_for_coeffs.fit(x_processed_linear, y_arr)
            best_score = model_for_coeffs.score(x_processed_linear, y_arr)
        best_model = model_for_coeffs # The model itself is the LinearRegression instance

    accuracy = best_score
    a_raw = best_model.intercept_ if hasattr(best_model, 'intercept_') else best_model.named_steps['linearregression'].intercept_
    b_raw = best_model.coef_ if hasattr(best_model, 'coef_') else best_model.named_steps['linearregression'].coef_

    if hasattr(b_raw, 'flatten'):
        b_flat = b_raw.flatten()
    else:
        b_flat = np.array([b_raw])

    num_coeffs = b_flat.shape[0]
    equation_parts = []

    if best_degree > 0: # Polynomial (single original feature)
        # Coefficients from b_flat correspond to feature names from final_poly_feature_names
        # e.g., names: ['x0', 'x0^2', 'x0^3'], coeffs: [c_for_x, c_for_x^2, c_for_x^3]
        for i in range(num_coeffs):
            coeff_val = b_flat[i]
            # Use feature name but replace 'x0' with 'x' for display
            # and handle powers, e.g., x0 -> x, x0^2 -> x^2
            feature_term = final_poly_feature_names[i].replace('x0', 'x')
            if feature_term == 'x^1': # if PolynomialFeatures outputs x0^1
                 feature_term = 'x'
            
            # For highest power term first in equation string (optional, current is ascending power from polyfeatures)
            # This loop iterates based on PolynomialFeatures output order (usually x, x^2, x^3)
            # To reverse for display: iterate b_flat and final_poly_feature_names in reverse.
            # For now, using PolyFeatures order: c1x + c2x^2 + ...
            
            # Standard display: c_n*x^n + ... + c_1*x. Coeffs are typically for x, x^2, ...
            # Let's stick to the order from PolynomialFeatures for simplicity of matching coeffs.
            # Term will be like "coeff*x", "coeff*x^2"
            current_term_display = feature_term

            if i == 0: # First coefficient term
                equation_parts.append(f"{round(coeff_val, 4)}{current_term_display}")
            else:
                sign = "+" if coeff_val >= 0 else "-"
                equation_parts.append(f"{sign} {round(abs(coeff_val), 4)}{current_term_display}")
    
    else: # Multiple Linear Regression (best_degree == 0)
        letters = list('abcdefghijklmnopqrstuvwxyz')
        reqletters = [letters[i % len(letters)] for i in range(num_coeffs)]
        if num_coeffs > 0:
            equation_parts.append(f"{round(b_flat[0], 4)}{reqletters[0]}")
            for i in range(1, num_coeffs):
                coeff_val = b_flat[i]
                sign = "+" if coeff_val >= 0 else "-"
                equation_parts.append(f"{sign} {round(abs(coeff_val), 4)}{reqletters[i]}")

    # Process and add intercept term
    if isinstance(a_raw, (np.ndarray, list)):
        mainvar_val = a_raw[0] if len(a_raw) > 0 else 0.0
    else:
        mainvar_val = a_raw
    mainvar_rounded = round(float(mainvar_val), 4)

    intercept_sign = "+" if mainvar_rounded >= 0 else "-"
    if not equation_parts: # Only intercept
        last = f"{mainvar_rounded}"
    else:
        equation_parts.append(f"{intercept_sign} {round(abs(mainvar_rounded), 4)}")
        last = " ".join(equation_parts)
        if last.startswith("+ "):
            last = last[2:]

    return f"f(x): {last}\naccuracy: {round(accuracy * 100, 2)}%"
