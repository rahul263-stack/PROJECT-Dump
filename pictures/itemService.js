clothesShop.factory('Item', ['$resource', function($resource) {

    return $resource('test/mock/clothes.json', {}, {
      query: {
        method: 'GET',
        isArray: true
      },
  });

}]);
