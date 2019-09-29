const { create } = require('./create');
const { remove } = require('./remove');
const { update } = require('./update');
const { get } = require('./get');

const package = module.exports = {};

package.create = create;
package.remove = remove;
package.update = update;
package.get = get;

return package;
