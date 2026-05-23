<img width="2337" height="1050" alt="image" src="https://github.com/user-attachments/assets/d869f97c-205b-4bc0-abc9-62ce50e905a4" />

## How to run the code
### first stage：train tail-classes teacher model
`pytohn AIBSD_teacher.py --config ./configs/isic/100/isic_AIBSD_teacher.yml`
### second stage：train full-task student model
`pytohn AIBSD.py --config ./configs/isic/100/isic_AIBSD.yml`
