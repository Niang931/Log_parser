// First start of `mongo` only (the entrypoint runs this on an empty data dir):
// one read-write user for the server, one read-only user for the Query page's free-form pipelines.
for (const k of ["MONGO_APP_USER", "MONGO_APP_PASSWORD", "MONGO_RO_USER", "MONGO_RO_PASSWORD"]) {
  if (!process.env[k]) throw new Error(`${k} is empty: set the MONGO_* passwords in .env`);
}
const name = process.env.MONGO_DB || "logpipe";
const app = db.getSiblingDB(name);
app.createUser({user: process.env.MONGO_APP_USER, pwd: process.env.MONGO_APP_PASSWORD,
                roles: [{role: "readWrite", db: name}]});
app.createUser({user: process.env.MONGO_RO_USER, pwd: process.env.MONGO_RO_PASSWORD,
                roles: [{role: "read", db: name}]});
