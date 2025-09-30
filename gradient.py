def gradient():
    print("Gradient")
    
    grad_x = []
    grad_y = []

    for i in range(rows):
        for j in range(cols):
            if j > 0 and j < cols - 1:
                sobel_x = gray[i][j + 1] - gray[i][j - 1]
            else:
                sobel_x = 0

            if i > 0 and i < rows - 1:
                sobel_y = gray[i + 1][j] - gray[i - 1][j]
            else:
                sobel_y = 0

            grad_x.append(sobel_x)
            grad_y.append(sobel_y)

    gradient_magnitude = np.sqrt(np.array(grad_x)**2 + np.array(grad_y)**2)
    # plt.imshow(gradient_magnitude.reshape(gray.shape), cmap="gray")
    # plt.title("Gradient Magnitude")
    # plt.axis("off")
    # plt.show()