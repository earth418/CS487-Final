/**
 * Author: Danny Rakita
 * Description: For CPSC-487-587 3D Spatial Modeling and Computing at Yale University
 */

import {
    add_matrix_matrix,
    frobenius_norm_matrix, identity_matrix, mul_matrix_matrix,
    mul_matrix_scalar,
    roll_list_into_column_vec_matrix, roll_list_into_row_vec_matrix, sub_matrix_matrix, transpose, unroll_matrix_to_list
} from "./utils_math.js";
import {minimize_Powell, minimize_GradientDescent, minimize_L_BFGS} from "./opt.js";

export function gradient_fd(f, x) {
    let f0 = f(x);

    let perturbation = 0.0001;

    let grad = [[]];

    for(let i=0; i<x.length; i++) {
        let xh = x.slice();
        xh[i][0] += perturbation;
        let fh = f(xh);
        let res = (fh - f0) / perturbation;
        grad[0].push(res);
    }

    return [f0, grad];
}

/*
export function optimization_gradient_descent(f, x0, lambda = 0.05, gradient_norm_tolerance=0.01, relative_tolerance=0.01, max_iter=0) {
    let x_star = 0;
    let f_star = 0;
    let gradient_norm = 0;
    let x_iterations = [];
    let f_iterations = [];
    let num_iters = 0;
    let prev_f = 0;
    let r = 0;

    let x = x0.slice();
    while (true) {
        num_iters++;

        let a = gradient_fd(f, x);
        let f_value = a[0];
        let grad = roll_list_into_column_vec_matrix(a[1]);
        let grad_norm = frobenius_norm_matrix(grad);

        x_iterations.push(x.slice());
        f_iterations.push(f_value);
        x_star = x.slice();
        f_star = f_value;
        gradient_norm = grad_norm;
        r = Math.abs(prev_f - f_value);

        if (max_iter !== 0) {if (num_iters >= max_iter) { break; }}
        if (grad_norm < gradient_norm_tolerance) { break; }
        if (num_iters > 1) { if (r < relative_tolerance) { break; } }

        let x_as_mat = roll_list_into_column_vec_matrix(x);
        let alpha = backtracking_line_search(f, x, transpose(mul_matrix_scalar(grad, -1.0)));
        let scaled_grad = mul_matrix_scalar(grad, -alpha);
        x = add_matrix_matrix(x_as_mat, transpose(scaled_grad));
        // x = unroll_matrix_to_list(x_iter);

        prev_f = f_value;
    }

    return new OptimizationResult(x_star, f_star, gradient_norm, num_iters, x_iterations, f_iterations, r);
}
*/

export function optimization_solve(f, x0, max_iter=100, solver='bfgs',) {
    if(solver === 'bfgs' || solver === 'BFGS') {
        return optimization_bfgs(f, x0, max_iter);
    } else if (solver === 'powell' || solver === 'POWELL') {
        return optimization_powell(f, x0, max_iter);
    } else if (solver === 'gd' || solver === 'GD') {
        return optimization_gradient_descent(f, x0, max_iter);
    }
}

export function optimization_powell(f, x0, max_iter=100) {
    x0 = unroll_matrix_to_list(x0);

    let solution = minimize_Powell(f, x0, max_iter);

    return solution.argument;
}

export function optimization_bfgs(f, x0, max_iter=100) {
    let g = function (x) {
        var grad = x.slice();
        var fx = f(x);
        var h = 1e-6; // step size

        for (var i = 0; i < x.length; i++) {

            // approximation using simple forward difference
            x[i] += h;
            var fxi = f(x);
            x[i] -= h;

            grad[i] = (fxi - fx) / h;
        }
        return grad;
    }

    let solution = minimize_L_BFGS(f, g, x0, max_iter);

    return solution.argument;
}

export function optimization_gradient_descent(f, x0, max_iter=100) {
    let g = function (x) {
        var grad = x.slice();
        var fx = f(x);
        var h = 1e-6; // step size

        for (var i = 0; i < x.length; i++) {

            // approximation using simple forward difference
            x[i] += h;
            var fxi = f(x);
            x[i] -= h;

            grad[i] = (fxi - fx) / h;
        }
        return grad;
    }

    let solution = minimize_GradientDescent(f, g, x0, max_iter);

    return solution.argument;
}

export function optimization_dummy(f, dimensions) {

}

export function backtracking_line_search(f, xk, pk, alpha_bar=5, c=0.95, phi=0.95) {
    let alpha = alpha_bar;

    let b = f(xk);

    let k = 0;
    while(true) {
        let a = f(add_matrix_matrix(xk, mul_matrix_scalar(pk, alpha)));
        let d = mul_matrix_scalar(mul_matrix_matrix(gradient_fd(f, xk)[1], pk), c*alpha)[0][0];
        let e = b + d;

        if (a <= e) { break; }
        else { alpha = phi*alpha; }

        k++;
        if(k>20) { break; }
    }

    return alpha;
}

export class OptimizationBFGS {
    constructor(f, problem_size) {
        this.f = f;
        this.problem_size = problem_size;
        this.I = identity_matrix(this.problem_size);
        this.Hxk = identity_matrix(this.problem_size);
    }

    optimize(x0, gradient_norm_tolerance=0.001, max_iter=100) {
        let x_iterations = [];
        let f_iterations = [];
        let xk = x0.slice();

        let k = 0;
        while (true) {
            let grad_result = gradient_fd(this.f, xk);
            let Df_xk = grad_result[1];
            x_iterations.push(xk);
            f_iterations.push(grad_result[0]);

            let pk = mul_matrix_matrix(mul_matrix_scalar(this.Hxk, -1.0), transpose(Df_xk));
            let alpha = backtracking_line_search(this.f, xk, pk);
            console.log(alpha);
            let xk1 = add_matrix_matrix(xk, mul_matrix_scalar(pk, alpha));

            let grad_result_2 = gradient_fd(this.f, xk1);
            let Df_xk1 = grad_result_2[1];
            let sk = sub_matrix_matrix(xk1, xk);
            let yk = sub_matrix_matrix(Df_xk1, Df_xk);

            let rhok_d = mul_matrix_matrix(yk, sk);
            let rhok = 1.0 / rhok_d[0][0];

            let a = sub_matrix_matrix(this.I, mul_matrix_scalar(mul_matrix_matrix(sk, yk), rhok));
            let b = sub_matrix_matrix(this.I, mul_matrix_scalar(mul_matrix_matrix(transpose(yk), transpose(sk)), rhok));
            let c = mul_matrix_scalar(mul_matrix_matrix(sk, transpose(sk)), rhok);
            let Hxk1 = add_matrix_matrix(mul_matrix_matrix(mul_matrix_matrix(a, this.Hxk), b), c);
            this.Hxk = Hxk1;

            xk = xk1;
            var gradient_norm = frobenius_norm_matrix(Df_xk);

            if (gradient_norm < gradient_norm_tolerance || k > max_iter) { break; }

            k++;
        }

        return new OptimizationResult(x_iterations[x_iterations.length-1], f_iterations[f_iterations.length-1], gradient_norm, k, x_iterations, f_iterations, 0.0);
    }
}

function dot_product(a, b) {
    return a.reduce((sum, current, i) => sum + (current * b[i]), 0);
}

export class OptimizationResult {
    constructor(x_star, f_star, gradient_norm, num_iters, x_iterations, f_iterations, relative_tolerance) {
        this.x_star = x_star;
        this.f_star = f_star;
        this.gradient_norm = gradient_norm;
        this.num_iters = num_iters;
        this.x_iterations = x_iterations;
        this.f_iterations = f_iterations;
        this.relative_tolerance = relative_tolerance;
    }
}