# DPO推导
**损失函数**
论文中定义的损失函数为:
$$
L_R(r_\phi,D) = -E_{(x,y_w,y_l)\sim D}[\log \sigma(r_{\phi}(x,y_w)-r_{\phi}(x,y_l))]\\
$$
其中$sigmoid$函数为$\sigma(z) = \frac{1}{1+e^{-z}}$，$D = \{(x^{(i)},y_w^{(i)},y_l^{(i)})\}_{i=1}^N$
**偏好函数**
论文的目的是使损失函数尽可能的小，人为定义偏好函数
$$p^*(y_1\succ y_2|x) = \frac{e^{r^*(x,y_1)}}{e^{r^*(x,y_1)}+e^{r^*(x,y_2)}} = \sigma(r^*(x,y_1)-r^*(x,y_2))$$
显然，$r^*(x,y_1)-r^*(x,y_2)$越大，偏好函数的值越大
对偏好函数做最大似然估计，并且取对数，得到$L(\theta) = \Sigma_{i=1}^N\log\sigma(r_\theta(x^{(i)},y_w^{(i)})-r_\theta(x^{(i)},y_l^{(i)}))$类似于损失函数的形式。
**RL阶段定义的目标（人为设定）**
$$
\max_\pi E_{x\sim D,y\sim\pi(·|x)}[r_\phi(x,y)]-\beta D_{KL}(\pi(y|x)||\pi_{ref}(y|x))\\
实质上就是求\max J(\pi) = \Sigma_y\pi(y|x)r(x,y)-\beta \pi(y|x)\log\frac{\pi(y|x)}{\pi_{ref}(y|x)}
$$
上述式子有归一化约束：$\Sigma_y\pi(y|x)=1$
**求解**
建立$Lagrange$函数:
$$
\mathbf{L} = \Sigma_y\pi(y|x)r(x,y)-\beta \pi(y|x)\log\frac{\pi(y|x)}{\pi_{ref}(y|x)}+\lambda(\Sigma_y\pi(y|x)-1)\\
对每个\pi(y_i|x)求偏导，\frac{\delta \mathbf{L}}{\delta \pi(y_i|x)} = r(x,y_i)-\beta(\log\frac{\pi(y_i|x)}{\pi_{ref}(y_i|x)}+1)+\lambda=0
\\又有\Sigma_y\pi_{ref}(y|x) = 1\\
得到\pi(y_i|x) = \frac{\pi_{ref}(y_i|x)e^\frac{r(x,y_i)}{\beta}}{Z(x)},Z(x) = \Sigma_y\pi_{ref}(y|x)e^\frac{r(x,y)}{\beta}
$$
**反解r(x,y)**
对上面求出来的式子左右两边取对数：
$$
\log \pi(y_i|x) = \log \pi_{ref}(y_i|x)+\frac{r(x,y_i)}{\beta}-\log Z(x)\\
r(x,y_i) = \beta\log\frac{\pi(y_i|x)}{\pi_{ref}(y_i|x)}+\beta\log Z(x)
$$
**综上，得证reward可被policy替代**
实际上可以略去$Z(x)$，因为在损失函数的计算中会被抵消