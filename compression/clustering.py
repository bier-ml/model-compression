if __name__ == '__main__':
    print(1)
    # response = requests.get("https://ultralytics.com/images/zidane.jpg")
    # img = Image.open(BytesIO(response.content))
    # benchmark = ExecutionTimeBenchmark()
    # yolo = YoloV5()
    #
    # model_params = yolo.model.state_dict()
    # X = []
    # for param_name in tqdm(model_params):
    #     if 'weight' in param_name:
    #         X.extend(model_params[param_name].flatten())
    #
    # X = np.array(X)
    # kmeans = KMeans(n_clusters=8, random_state=0, n_init="auto").fit(X.reshape(-1, 1))
    # centers = kmeans.cluster_centers_.flatten()
    # centers = {i: center for i, center in enumerate(centers)}
    # labels = kmeans.labels_.flatten()
    #
    # cnt = 0
    # for param_name in tqdm(model_params):
    #     if 'weight' in param_name:
    #         params = model_params[param_name]
    #         initial_shape = params.shape
    #         arr = []
    #         for param in params.flatten():
    #             arr.append(centers[labels[cnt]])
    #             cnt += 1
    #         new_params = torch.from_numpy(np.array(arr)).reshape(initial_shape)
    #         print(((params - new_params) / params).sum() * 100)
    #         model_params[param_name] = torch.from_numpy(np.array(arr)).reshape(initial_shape)
    #
    # yolo.model.load_state_dict(model_params)
    #
    # res = yolo.sample_infer()
    # print(res)
