let imgsrc = game.scenes.get(game.user.viewedScene).data.img

// let img = new Image()
// img.src = imgsrc

let texture = await loadTexture(imgsrc)
// console.log(texture)

let sprite = new PIXI.Sprite(texture)
// console.log(sprite)

let app = new PIXI.Application()

let raw = app.renderer.extract.pixels(sprite)

let [w,h] = [sprite.width,sprite.height]

let data = []

console.log(sprite)

for (let i=0; i<w; i++){
	let row = []
	for (let j=0; j<h; j++){
		let ind = i*w+j*4
		row.push(raw.slice(ind,ind+4))
	}
	data.push(row)
}

console.log(data)