clothesShop.controller('mainController', ["Item", function(Item) {

  var self = this;
  self.cart = [];
  self.discount = 0;
  self.outOfOrder = false;
  self.list = true;
  self.voucherCodes = ['womenwear', 'casual', 'footwear'];
  self.clothesList = Item.query();

  self.addItem = function(item){

    itemSelected = {
      category: item.category,
      name: item.name,
      price: item.price,
      quantity: 1,
      image: item.image
    };

    if (item.quantity === 0) {
      self.outOfOrder = true;
    } else if (self.isNotInCart(item)) {
      self.addItemSelected(item, itemSelected);
    } else {
      self.addQuantity(item);
    }
  };

  self.isNotInCart = function(item) {
    for(var i = 0; i < self.cart.length; i++) {
      if (item.name === self.cart[i].name) {
        return false;
      }
    }
    return true;
  };

  self.addItemSelected = function(item, itemSelected){
    item.quantity --;
    self.cart.push(itemSelected);
    self.outOfOrder = false;
  };

  self.addQuantity = function(item){
    var index = self.cartIndex(item);
    self.cart[index].quantity ++;
    item.quantity --;
  };

  self.cartIndex = function(item) {
    for (var i = self.cart.length -1; i >=0; i--) {
      if (self.cart[i].name === item.name) {
        return i;
      }
    }
  };

  self.sum = function(){
    var tot = 0;
    for (var i = 0; i < self.cart.length; i++) {
      tot += (self.cart[i].price * self.cart[i].quantity);
    }
    return tot;
  };

  self.isFootwear = function(){
    for (var i = 0; i < self.cart.length; i++) {
      if(self.cart[i].category.includes('Footwear')) {
        return true;
      }
    }
  };

    self.isVoucherCorrect = function(){
      var prova = new RegExp(self.voucher);
      if (prova.test(self.voucherCodes) === true) {
        self.showOkVoucher();
        return true;
      } else {
        self.showWrongVoucher();
        return false;
      }
    };

    self.checkDiscount = function(){
      var tot = self.sum();
      if(self.isVoucherCorrect()) {
        if (tot > 75 && self.isFootwear()) {
          self.discount = 15;
        } else if (tot > 50) {
          self.discount = 10;
        } else {
          self.discount = 5;
        }
      }
      self.afterVoucher = true;
      return self.discount;
    };

    self.afterDisc = function(){
      var sum = self.sum();
      var total = sum - self.discount;
      return total;
    };

    self.countQuantity = function(){
      var totItems = 0;
      for(var i = 0; i < self.cart.length; i++) {
        totItems +=  self.cart[i].quantity;
      }
      return totItems;
    };

    self.showOkVoucher = function(){
      self.okVoucher = true;
      self.wrongVoucher = false;
    };

    self.showWrongVoucher = function(){
      self.okVoucher = false;
      self.wrongVoucher = true;
    };

    self.toCart = function(){
      self.checkout = !self.checkout;
      self.list = !self.list;
    };
  }]);

