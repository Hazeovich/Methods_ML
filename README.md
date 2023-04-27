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
    1. Линейная функция - $ h_\theta(x)=\theta_1x_1 + \theta_2x_2 + ... + \theta_nx_n $
    2. MSE - $ J(\theta)=\frac{1}{M}\sum\limits_{i=1}^M{(y_i-h_\theta(X_i))^2} $
    3. Градиент функции MSE - $ 
    \nabla J(\theta)=\{
        \frac{\partial{J}}{\partial\theta_1},
        \frac{\partial{J}}{\partial\theta_2},
        ...,
        \frac{\partial{J}}{\partial\theta_n} 
        \} 
    $
- Поиск линейной регрессии 
- **Метод наимешьних квадратов** квадратичная зависимость
- Lasso
    1. $ f(x)=a\cdot x+b \approx y$, $ |\{ a_i|a_i\in a,a_i=0\}|=k, 0<k\leq |a| = m $
    2. $ R^2=1-\frac{\sum\limits_{i=1}^n{(y_i-f(X_i))^2}}{\sum\limits_{i=1}^n{(y_i-\bar{y})^2}} \geq s$
## Logistic regression
- Реализация логистической регрессии
    1. $ f(x,\theta) = \sigma ( \sum\limits_{i=1}^n{x_i\theta_i} ) $
    2. $ \sigma(x)=\frac{1}{1+e^{-x}} $
- Реализация регрессии пуассона
    1. $ f(x, \theta) = \exp(\sum\limits_{i=1}^n\theta_ix_i + \theta_0) $
    2. $l(X, y, \theta) = \frac{1}{m}\sum\limits_{i=1}^n(y_i\log\frac{y_i}{f(X_i, \theta)} - y_i + f(X_i, \theta)) + \frac{\alpha}{2}\sum\limits_{i=1}^n\theta_i^2$
    3. $D^2 = 1 - \frac{D(y, \hat{y})}{D(y, \overline{y})}$
    4. $D(y, \hat{y}) = 2(y\log\frac{y}{\hat{y}} - y + \hat{y})$
    5. $\frac{\partial{l}}{\partial\theta_k} = \frac{1}{m}{\sum\limits_{i=1}^m(X_{i,k}(e^{\sum\limits_{j=1}^n\theta_jX_{i,j} + \theta_0} - y_i))+\alpha\theta_k},k=1...n$
    6. $ \frac{\partial{l}}{\partial\theta_0} = \frac{1}{m}{\sum\limits_{i=1}^m(e^{\sum\limits_{j=1}^n\theta_jX_{i,j}+ \theta_0} - y_i)},k=0$
    7. $X=\begin{vmatrix}
X_{1,0} & X_{1,1} & ... & X_{1,n}\\
X_{2,0} & X_{2,1} & ... & X_{2,n}\\
... & ... & ... & ...\\
X_{m,0} & X_{1,1} & ... & X_{m,n}\\
\end{vmatrix};
X_{1...m,0}=1;
\theta=\begin{vmatrix}
\theta_0 & \theta_1 & ... & \theta_n
\end{vmatrix};
y=\begin{vmatrix}
y_1 \\ y_2 \\ ... \\ y_m
\end{vmatrix};
\alpha=\begin{vmatrix}
\alpha_0 \\ \alpha_1 \\ ... \\ \alpha_n
\end{vmatrix};
\alpha_0 = 0
$	
    8. $\frac{\partial{l}}{\partial\theta_k} = \frac{1}{m}{(X(e^{\theta{X}} - y_i))+{\alpha}{\theta}}$
    9. *ADAM*
        1. $m_t = {\beta_1}{m_{t-1}} + (1-\beta_1)grad_t$
        1. $v_t = {\beta_2}{v_{t-1}} + (1-\beta_2)grad_t^2$
        1. $\hat{m_t}=\frac{m_t}{1-\beta_1^t}$
        1. $\hat{v_t}=\frac{v_t}{1-\beta_2^t}$
        1. $\theta_{t+1}=\theta_t - \frac{\eta}{\sqrt{\hat{v_t}}+\epsilon}\hat{m_t} $


