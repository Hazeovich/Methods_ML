# Methods_ML
## OneRule and Extrapolation
1. Реализация OneRule классификатора
    + Алгоритм OneR
    + Для каждого предиктора,
        + Для каждого значения этого предиктора составить правило следующим образом;
            + Подсчитать, как часто встречается каждое значение цели (класса)
            + Найти наиболее часто встречающийся класс
            + Составьте правило, присваивающее этот класс данному значению предиктора

        + Вычислите суммарную ошибку правил каждого предиктора

    + Выберите предиктор с наименьшей суммарной ошибкой.

2. Экстраполяция за счет подбора исходной функции
## Linear regression
- Функции обработки для линейной регрессии
    - Линейная функция - $h_\theta(x)=\theta_1x_1 + \theta_2x_2 + ... + \theta_nx_n$
    - MSE - $J(\theta)=\frac{1}{M}\sum\limits_{i=1}^M{(y_i-h_\theta(X_i))^2}$
    - Градиент функции MSE - $\nabla J(\theta)=\{\frac{\partial{J}}{\partial\theta_1},\frac{\partial{J}}{\partial\theta_2},...,\frac{\partial{J}}{\partial\theta_n}\}$
- Поиск линейной регрессии 
- **Метод наимешьних квадратов** квадратичная зависимость
- Lasso
    - $f(x)=a\cdot x+b \approx y$, $ |\{ a_i|a_i\in a,a_i=0\}|=k, 0<k\leq |a| = m$
    - $R^2=1-\frac{\sum\limits_{i=1}^n{(y_i-f(X_i))^2}}{\sum\limits_{i=1}^n{(y_i-\bar{y})^2}}\geq{s}$
## Logistic regression
- Реализация логистической регрессии
    - $f(x,\theta) = \sigma ( \sum\limits_{i=1}^n{x_i\theta_i} )$
    - $\sigma(x)=\frac{1}{1+e^{-x}}$
- Реализация регрессии пуассона
    - $f(x, \theta) = \exp(\sum\limits_{i=1}^n\theta_ix_i + \theta_0)$
    - $l(X, y, \theta) = \frac{1}{m}\sum\limits_{i=1}^n(y_i\log\frac{y_i}{f(X_i, \theta)} - y_i + f(X_i, \theta)) + \frac{\alpha}{2}\sum\limits_{i=1}^n\theta_i^2$
    - $D^2 = 1 - \frac{D(y, \hat{y})}{D(y, \overline{y})}$
    - $D(y, \hat{y}) = 2(y\log\frac{y}{\hat{y}} - y + \hat{y})$
    - $\frac{\partial{l}}{\partial\theta_k} = \frac{1}{m}{\sum\limits_{i=1}^m(X_{i,k}(e^{\sum\limits_{j=1}^n\theta_jX_{i,j} + \theta_0} - y_i))+\alpha\theta_k},k=1...n$
    - $\frac{\partial{l}}{\partial\theta_0} = \frac{1}{m}{\sum\limits_{i=1}^m(e^{\sum\limits_{j=1}^n\theta_jX_{i,j}+ \theta_0} - y_i)},k=0$	
    - $\frac{\partial{l}}{\partial\theta_k} = \frac{1}{m}{(X(e^{\theta{X}} - y_i))+{\alpha}{\theta}}$
    - *ADAM*
        - $m_t = {\beta_1}{m_{t-1}} + (1-\beta_1)grad_t$
        - $v_t = {\beta_2}{v_{t-1}} + (1-\beta_2)grad_t^2$
        - $\hat{m_t}=\frac{m_t}{1-\beta_1^t}$
        - $\hat{v_t}=\frac{v_t}{1-\beta_2^t}$
        - $\theta_{t+1}=\theta_t - \frac{\eta}{\sqrt{\hat{v_t}}+\epsilon}\hat{m_t} $

$$X=\left\lbrack \matrix{ \
    X_{1,0}&X_{1,1}&...&X_{1,n} \cr \
    X_{2,0}&X_{2,1}&...&X_{2,n} \cr \
    ...&...&...&... \cr \
    X_{m,0}&X_{1,1}&...&X_{m,n} \
} \right\rbrack;
X_{1...m,0}=1;
\theta=\left\lbrack \matrix{ \
    theta_0 & \theta_1 & ... & \theta_n     
} \right\rbrack;
y=\left\lbrack \matrix{ \
    y_1 \cr y_2 \cr ... \cr y_m \
} \right\rbrack;
\alpha=\left\lbrack \matrix{ \
    \alpha_0 \cr \alpha_1 \cr ... \cr \alpha_n \
} \right\rbrack;
\alpha_0 = 0
$$


